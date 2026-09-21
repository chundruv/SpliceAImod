"""Exactness of the unpadded / conv2d fused-ensemble paths vs the padded graph.

Uses the shipped weights. Deviations must be at fp32 reassociation level.
"""
import os
import torch
import pytest

from spliceai.batch.fused_inference import FusedSpliceAIEnsemble
from spliceai.batch.fused_valid import wrap_conv_impl, CONV_IMPLS

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'spliceai', 'models')


@pytest.fixture(scope='module')
def fused():
    sds = [torch.load(os.path.join(MODEL_DIR, f'spliceai{i}.pt'), map_location='cpu', weights_only=True)
           for i in range(1, 6)]
    m = FusedSpliceAIEnsemble()
    m.load_from_state_dicts([{k: v.float() for k, v in s.items()} for s in sds])
    return m.eval()


def _onehot(n, L, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.zeros(n, 4, L)
    idx = torch.randint(0, 4, (n, L), generator=g)
    x.scatter_(1, idx[:, None, :], 1.0)
    x[0, :, :250] = 0.0          # a run of N's, as transcript-boundary padding produces
    return x


@pytest.mark.parametrize('L', [11001, 10007, 12001])   # -D 500, odd length (pad path), -D 1000
@pytest.mark.parametrize('impl', [i for i in CONV_IMPLS if i != 'padded'])
def test_matches_padded(fused, impl, L):
    x = _onehot(2, L)
    with torch.no_grad():
        ref = fused(x)
        out = wrap_conv_impl(fused, impl).eval()(x)
    assert out.shape == ref.shape == (2, 3, L - fused.CL)
    assert (out - ref).abs().max().item() < 1e-5
