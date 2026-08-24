#!/usr/bin/env python
"""Assert every shipped spliceai{i}.pt tensor equals its spliceai{i}.h5 source.

Addresses review finding 13: the existing suite tests PyTorch against PyTorch
only, which is why the initial_skip mis-mapping (conv1d_11 loaded in place of
conv1d_2, in all five released checkpoints) was never caught.

This test needs h5py and torch, NOT TensorFlow: it reads the Keras HDF5
weight groups directly and compares them to the state dict under the
(2, 1, 0) kernel transpose. Tolerance is exact -- conversion is a transpose
and a dtype cast, not a computation, so any nonzero difference is a wiring bug.

Also checks:
  * every Keras conv/BN layer is consumed exactly once (a layer used twice is
    the signature of the initial_skip defect);
  * the five checkpoints are pairwise distinct (a copy-paste in the conversion
    loop would silently ship five identical models, destroying the ensemble).

Run: python tests/test_keras_equivalence.py
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

MODELS_DIR = os.path.join(ROOT, 'spliceai', 'models')

_failures = []
_checks = 0


def check(label, ok, detail=''):
    global _checks
    _checks += 1
    if ok:
        print(f"[PASS] {label}")
    else:
        print(f"[FAIL] {label}  {detail}")
        _failures.append(label)


def layer_index(name):
    """Trailing-integer sort key -- must match convert_keras_to_pytorch."""
    tail = name.rsplit('_', 1)[-1]
    return int(tail) if tail.isdigit() else 0


def read_keras_weights(h5_path):
    """{layer_name: [array, ...]} from a Keras HDF5 weights file."""
    import h5py
    out = {}
    with h5py.File(h5_path, 'r') as f:
        grp = f['model_weights'] if 'model_weights' in f else f

        def collect(name, obj):
            if isinstance(obj, h5py.Dataset):
                top = name.split('/')[0]
                out.setdefault(top, []).append((name, np.array(obj)))

        grp.visititems(collect)
    # Keras stores weight_names order per layer; sort by dataset name so
    # kernel precedes bias and gamma/beta/mean/var stay in Keras order.
    ordered = {}
    for layer, items in out.items():
        def wkey(pair):
            n = pair[0].lower()
            for i, tok in enumerate(('kernel', 'gamma', 'beta',
                                     'moving_mean', 'moving_variance', 'bias')):
                if tok in n:
                    # bias must follow kernel; gamma/beta/mean/var in BN order
                    return {'kernel': 0, 'bias': 1, 'gamma': 0, 'beta': 1,
                            'moving_mean': 2, 'moving_variance': 3}[tok]
            return 9
        ordered[layer] = [a for _, a in sorted(items, key=wkey)]
    return ordered


#: The 4 skip convolutions are not a separate `skips` ModuleList: they are
#: interleaved into residual_units at these slots (verified against the shipped
#: state dicts, where residual_units.{4,9,14,19} carry a bare `conv` and no
#: batchnorm, while the other 16 slots carry conv1/conv2/batchnorm1/batchnorm2).
SKIP_SLOTS = frozenset({4, 9, 14, 19})
N_RESIDUAL_SLOTS = 20


def build_expected_mapping(conv_layers, bn_layers):
    """Reproduce the converter's positional walk as an explicit name mapping.

    Returns {pytorch_param_prefix: keras_layer_name}. Derived from the state
    dict's own structure and the Keras graph order rather than by re-reading
    convert_keras_to_pytorch, so this is a genuine cross-check of the wiring.

    Architecture (SpliceAI-10k): initial_conv, initial_skip.conv, then 20
    residual_units slots of which 16 are true residual units (2 convs + 2 BNs)
    and 4 are skip convs, then the final conv.
    1 + 1 + 32 + 4 + 1 = 39 convs, 16 * 2 = 32 BNs.
    """
    m = {}
    ci = bi = 0
    m['initial_conv'] = conv_layers[ci]; ci += 1
    m['initial_skip.conv'] = conv_layers[ci]; ci += 1
    for slot in range(N_RESIDUAL_SLOTS):
        if slot in SKIP_SLOTS:
            m[f'residual_units.{slot}.conv'] = conv_layers[ci]; ci += 1
        else:
            m[f'residual_units.{slot}.batchnorm1'] = bn_layers[bi]; bi += 1
            m[f'residual_units.{slot}.batchnorm2'] = bn_layers[bi]; bi += 1
            m[f'residual_units.{slot}.conv1'] = conv_layers[ci]; ci += 1
            m[f'residual_units.{slot}.conv2'] = conv_layers[ci]; ci += 1
    m['final_conv'] = conv_layers[ci]; ci += 1
    return m, ci, bi


def main():
    import torch

    print("=" * 68)
    print("Keras (.h5) vs PyTorch (.pt) checkpoint equivalence")
    print("=" * 68)

    h5s = [os.path.join(MODELS_DIR, f'spliceai{i}.h5') for i in range(1, 6)]
    pts = [os.path.join(MODELS_DIR, f'spliceai{i}.pt') for i in range(1, 6)]
    missing = [p for p in h5s + pts if not os.path.exists(p)]
    if missing:
        print(f"SKIP: missing checkpoint file(s): "
              f"{', '.join(os.path.basename(p) for p in missing)}")
        return 0

    conv_ref = None
    state_dicts = []

    for i, (h5_path, pt_path) in enumerate(zip(h5s, pts), start=1):
        print(f"\n--- spliceai{i} ---")
        kw = read_keras_weights(h5_path)
        conv_layers = sorted([k for k in kw if k.startswith('conv1d')],
                             key=layer_index)
        bn_layers = sorted([k for k in kw if k.startswith('batch_normalization')],
                           key=layer_index)
        check(f"m{i}: 39 conv + 32 BN layers in .h5",
              len(conv_layers) == 39 and len(bn_layers) == 32,
              f"got {len(conv_layers)} conv, {len(bn_layers)} bn")
        if len(conv_layers) != 39 or len(bn_layers) != 32:
            continue

        sd = torch.load(pt_path, map_location='cpu', weights_only=True)
        state_dicts.append(sd)

        mapping, ci, bi = build_expected_mapping(conv_layers, bn_layers)
        check(f"m{i}: mapping consumes every conv and BN layer exactly once",
              ci == 39 and bi == 32 and
              len(set(mapping.values())) == len(mapping),
              f"conv_used={ci} bn_used={bi} unique={len(set(mapping.values()))}"
              f"/{len(mapping)}")

        n_tensor_checks = 0
        worst = ('', 0.0)
        for prefix, kname in sorted(mapping.items()):
            arrays = kw[kname]
            if 'batchnorm' in prefix:
                pairs = [('weight', 0), ('bias', 1),
                         ('running_mean', 2), ('running_var', 3)]
                for pname, aidx in pairs:
                    key = f'{prefix}.{pname}'
                    if key not in sd or aidx >= len(arrays):
                        check(f"m{i}: {key} present", False, "missing")
                        continue
                    got = sd[key].numpy()
                    exp = arrays[aidx]
                    d = float(np.abs(got - exp).max()) if got.shape == exp.shape else float('inf')
                    if d > worst[1]:
                        worst = (key, d)
                    n_tensor_checks += 1
                    if d != 0.0:
                        check(f"m{i}: {key} == {kname}", False,
                              f"max|diff|={d:.3e} shapes {got.shape} vs {exp.shape}")
            else:
                # conv: Keras (k, in, out) -> PyTorch (out, in, k)
                wkey, bkey = f'{prefix}.weight', f'{prefix}.bias'
                if wkey not in sd:
                    check(f"m{i}: {wkey} present", False, "missing")
                    continue
                got = sd[wkey].numpy()
                exp = np.transpose(arrays[0], (2, 1, 0))
                d = float(np.abs(got - exp).max()) if got.shape == exp.shape else float('inf')
                if d > worst[1]:
                    worst = (wkey, d)
                n_tensor_checks += 1
                if d != 0.0:
                    check(f"m{i}: {wkey} == {kname}.kernel", False,
                          f"max|diff|={d:.3e} shapes {got.shape} vs {exp.shape}")
                if bkey in sd and len(arrays) > 1:
                    gb, eb = sd[bkey].numpy(), arrays[1]
                    db = float(np.abs(gb - eb).max()) if gb.shape == eb.shape else float('inf')
                    if db > worst[1]:
                        worst = (bkey, db)
                    n_tensor_checks += 1
                    if db != 0.0:
                        check(f"m{i}: {bkey} == {kname}.bias", False,
                              f"max|diff|={db:.3e}")

        check(f"m{i}: all {n_tensor_checks} tensors match .h5 exactly",
              worst[1] == 0.0,
              f"worst: {worst[0]} max|diff|={worst[1]:.3e}")

        if conv_ref is None:
            conv_ref = conv_layers

    print("\n--- Ensemble distinctness ---")
    if len(state_dicts) == 5:
        probe = 'final_conv.weight'
        ok = True
        for a in range(5):
            for b in range(a + 1, 5):
                if np.array_equal(state_dicts[a][probe].numpy(),
                                  state_dicts[b][probe].numpy()):
                    check(f"m{a+1} and m{b+1} differ on {probe}", False,
                          "identical -- ensemble is degenerate")
                    ok = False
        check(f"all five checkpoints pairwise distinct on {probe}", ok)

    print("\n" + "=" * 68)
    if _failures:
        print(f"{len(_failures)} of {_checks} checks FAILED:")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print(f"All {_checks} checks passed.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
