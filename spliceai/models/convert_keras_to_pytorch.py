#!/usr/bin/env python3
"""
Convert Keras SpliceAI models to PyTorch format.

Usage:
    python convert_keras_to_pytorch.py

This will convert all spliceai*.h5 or spliceai*.keras files in the current directory
to PyTorch .pt format.
"""

import os
import sys
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================
# PyTorch SpliceAI Model Definition
# ============================================

class ResidualUnit(nn.Module):
    """Residual unit with dilated convolutions"""
    
    def __init__(self, l, w, ar):
        super().__init__()
        self.batchnorm1 = nn.BatchNorm1d(l)
        self.batchnorm2 = nn.BatchNorm1d(l)
        padding = (w - 1) * ar // 2
        self.conv1 = nn.Conv1d(l, l, w, dilation=ar, padding=padding)
        self.conv2 = nn.Conv1d(l, l, w, dilation=ar, padding=padding)

    def forward(self, x, skip):
        out = self.conv1(F.relu(self.batchnorm1(x)))
        out = self.conv2(F.relu(self.batchnorm2(out)))
        return x + out, skip


class Skip(nn.Module):
    """Skip connection with 1x1 convolution"""
    
    def __init__(self, l):
        super().__init__()
        self.conv = nn.Conv1d(l, l, 1)

    def forward(self, x, skip):
        return x, self.conv(x) + skip


class SpliceAI(nn.Module):
    """SpliceAI model architecture in PyTorch"""
    
    def __init__(self, L=32, W=None, AR=None):
        super(SpliceAI, self).__init__()
        
        if W is None:
            W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
                           21, 21, 21, 21, 41, 41, 41, 41])
        if AR is None:
            AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
                            10, 10, 10, 10, 25, 25, 25, 25])
        
        self.L = L
        self.W = W
        self.AR = AR
        self.CL = 2 * np.sum(AR * (W - 1))
        
        self.initial_conv = nn.Conv1d(4, L, 1)
        self.initial_skip = Skip(L)
        
        self.residual_units = nn.ModuleList()
        for i, (w, r) in enumerate(zip(W, AR)):
            self.residual_units.append(ResidualUnit(L, w, r))
            if (i + 1) % 4 == 0:
                self.residual_units.append(Skip(L))
        
        self.final_conv = nn.Conv1d(L, 3, 1)

    def forward(self, x):
        # Store original length for cropping calculation
        CL = self.CL
        
        x = self.initial_conv(x)
        x, skip = self.initial_skip(x, 0)
        
        for m in self.residual_units:
            x, skip = m(x, skip)
        
        # Crop to remove context
        crop_left = CL // 2
        crop_right = CL // 2
        if crop_right > 0:
            skip = skip[:, :, crop_left:-crop_right]
        else:
            skip = skip[:, :, crop_left:]
            
        out = self.final_conv(skip)
        return F.softmax(out, dim=1)


def create_spliceai_model(flanking_size=10000):
    """Create a SpliceAI model with appropriate hyperparameters"""
    L = 32
    
    if flanking_size == 80:
        W = np.asarray([11, 11, 11, 11])
        AR = np.asarray([1, 1, 1, 1])
    elif flanking_size == 400:
        W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11])
        AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4])
    elif flanking_size == 2000:
        W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
                       21, 21, 21, 21])
        AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
                        10, 10, 10, 10])
    else:  # 10000
        W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
                       21, 21, 21, 21, 41, 41, 41, 41])
        AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
                        10, 10, 10, 10, 25, 25, 25, 25])
    
    model = SpliceAI(L, W, AR)
    model.eval()
    return model


def convert_keras_weights(keras_path, pytorch_model):
    """
    Convert Keras weights to PyTorch model.
    
    Keras SpliceAI architecture:
    - conv1d (initial): 4 -> 32 channels, kernel 1
    - conv1d_1 (skip): 32 -> 32, kernel 1  
    - Then alternating residual blocks and skip connections
    - Final conv1d: 32 -> 3, kernel 1
    
    Keras layer naming: conv1d, conv1d_1, conv1d_2, ...
    BatchNorm: batch_normalization, batch_normalization_1, ...
    """
    try:
        import h5py
    except ImportError:
        print("ERROR: h5py required for Keras conversion. Install with: pip install h5py")
        return False
    
    # Try loading with keras first (handles .keras format better)
    weights_dict = {}
    
    try:
        # Try tensorflow.keras
        import tensorflow as tf
        print(f"Loading {keras_path} with TensorFlow...")
        keras_model = tf.keras.models.load_model(keras_path, compile=False)
        
        # Extract weights by layer name
        for layer in keras_model.layers:
            layer_weights = layer.get_weights()
            if layer_weights:
                weights_dict[layer.name] = layer_weights
                print(f"  Layer {layer.name}: {[w.shape for w in layer_weights]}")
        
    except Exception as e:
        print(f"TensorFlow load failed: {e}")
        print("Trying h5py direct access...")
        
        try:
            with h5py.File(keras_path, 'r') as f:
                # Print structure for debugging
                def print_item(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        print(f"  {name}: {obj.shape}")
                    return None
                
                print("H5 file structure:")
                f.visititems(print_item)
                
                # Try to find weights
                if 'model_weights' in f:
                    weights_group = f['model_weights']
                else:
                    weights_group = f
                
                # Extract weights
                for layer_name in weights_group.keys():
                    try:
                        layer = weights_group[layer_name]
                        if layer_name in layer:
                            layer = layer[layer_name]
                        
                        # Order these the way Keras' layer.get_weights() does,
                        # because get_conv_weights/get_bn_weights below index
                        # them positionally: conv -> [kernel, bias];
                        # BN -> [gamma, beta, moving_mean, moving_variance].
                        # Plain sorted(layer.keys()) gives ['bias:0','kernel:0']
                        # and ['beta:0','gamma:0','moving_mean:0',
                        # 'moving_variance:0'], i.e. kernel/bias swapped and
                        # gamma/beta swapped -- the conv case raises
                        # "axes don't match array" on the (32,) bias and the BN
                        # case would silently load beta as gamma.
                        _WEIGHT_ORDER = ('kernel', 'gamma', 'beta',
                                         'moving_mean', 'moving_variance',
                                         'bias')

                        def _weight_rank(key):
                            stem = key.split(':')[0]
                            try:
                                return (_WEIGHT_ORDER.index(stem), key)
                            except ValueError:
                                return (len(_WEIGHT_ORDER), key)

                        layer_weights = []
                        for key in sorted(layer.keys(), key=_weight_rank):
                            layer_weights.append(np.array(layer[key]))
                        
                        if layer_weights:
                            weights_dict[layer_name] = layer_weights
                            print(f"  H5 Layer {layer_name}: {[w.shape for w in layer_weights]}")
                    except Exception as e2:
                        print(f"  Could not extract {layer_name}: {e2}")
                        
        except Exception as e2:
            print(f"H5 direct access also failed: {e2}")
            return False
    
    if not weights_dict:
        print("ERROR: Could not extract any weights from Keras model")
        return False
    
    # Map Keras weights to PyTorch model
    print("\nMapping weights to PyTorch model...")
    
    # Find conv and bn layers by pattern.
    #
    # These lists are consumed positionally below (conv_idx/bn_idx walk them in
    # order), so the sort MUST reproduce the order the layers appear in the
    # Keras graph. Plain string sort does NOT: it orders
    # conv1d_1, conv1d_10, conv1d_11, ..., conv1d_2, ...
    # so index 1 lands on conv1d_10 rather than conv1d_2 and every subsequent
    # index is displaced. Sort on the trailing integer instead -- verified
    # against the stored model_config for all five released models: the
    # numeric order is exactly the graph's topological order.
    def _layer_index(name):
        tail = name.rsplit('_', 1)[-1]
        return int(tail) if tail.isdigit() else 0

    conv_layers = sorted([k for k in weights_dict.keys() if k.startswith('conv1d')],
                         key=_layer_index)
    bn_layers = sorted([k for k in weights_dict.keys() if k.startswith('batch_normalization')],
                       key=_layer_index)

    print(f"Found {len(conv_layers)} conv layers: {conv_layers[:5]}...")
    print(f"Found {len(bn_layers)} bn layers: {bn_layers[:5]}...")
    
    # Expected layer counts for SpliceAI-10k:
    # - 1 initial conv
    # - 1 initial skip conv
    # - 16 residual units * 2 convs = 32 convs
    # - 4 skip convs (after every 4 residual units)
    # - 1 final conv
    # Total: 1 + 1 + 32 + 4 + 1 = 39 conv layers
    # Total BN: 32 (2 per residual unit)
    
    conv_idx = 0
    bn_idx = 0
    
    def get_conv_weights(idx):
        """Get conv weights, converting from Keras to PyTorch format"""
        if idx >= len(conv_layers):
            return None, None
        name = conv_layers[idx]
        w = weights_dict[name]
        # Keras conv1d kernel: (kernel_size, in_channels, out_channels)
        # PyTorch conv1d weight: (out_channels, in_channels, kernel_size)
        kernel = np.transpose(w[0], (2, 1, 0))
        bias = w[1] if len(w) > 1 else np.zeros(kernel.shape[0])
        return torch.from_numpy(kernel.copy()).float(), torch.from_numpy(bias.copy()).float()
    
    def get_bn_weights(idx):
        """Get batch norm weights"""
        if idx >= len(bn_layers):
            return None
        name = bn_layers[idx]
        w = weights_dict[name]
        # Keras BN: [gamma, beta, moving_mean, moving_variance]
        # PyTorch BN: weight (gamma), bias (beta), running_mean, running_var
        if len(w) >= 4:
            return {
                'weight': torch.from_numpy(w[0].copy()).float(),
                'bias': torch.from_numpy(w[1].copy()).float(),
                'running_mean': torch.from_numpy(w[2].copy()).float(),
                'running_var': torch.from_numpy(w[3].copy()).float(),
            }
        return None
    
    # Transfer initial conv
    kernel, bias = get_conv_weights(conv_idx)
    if kernel is not None:
        pytorch_model.initial_conv.weight.data = kernel
        pytorch_model.initial_conv.bias.data = bias
        print(f"  initial_conv: {kernel.shape}")
    conv_idx += 1
    
    # Transfer initial skip conv
    kernel, bias = get_conv_weights(conv_idx)
    if kernel is not None:
        pytorch_model.initial_skip.conv.weight.data = kernel
        pytorch_model.initial_skip.conv.bias.data = bias
        print(f"  initial_skip.conv: {kernel.shape}")
    conv_idx += 1
    
    # Transfer residual units and skip connections
    residual_idx = 0
    skip_idx = 0
    
    for i, module in enumerate(pytorch_model.residual_units):
        if isinstance(module, ResidualUnit):
            # Get BN1 weights
            bn_w = get_bn_weights(bn_idx)
            if bn_w:
                module.batchnorm1.weight.data = bn_w['weight']
                module.batchnorm1.bias.data = bn_w['bias']
                module.batchnorm1.running_mean.data = bn_w['running_mean']
                module.batchnorm1.running_var.data = bn_w['running_var']
            bn_idx += 1
            
            # Get conv1 weights
            kernel, bias = get_conv_weights(conv_idx)
            if kernel is not None:
                module.conv1.weight.data = kernel
                module.conv1.bias.data = bias
            conv_idx += 1
            
            # Get BN2 weights
            bn_w = get_bn_weights(bn_idx)
            if bn_w:
                module.batchnorm2.weight.data = bn_w['weight']
                module.batchnorm2.bias.data = bn_w['bias']
                module.batchnorm2.running_mean.data = bn_w['running_mean']
                module.batchnorm2.running_var.data = bn_w['running_var']
            bn_idx += 1
            
            # Get conv2 weights
            kernel, bias = get_conv_weights(conv_idx)
            if kernel is not None:
                module.conv2.weight.data = kernel
                module.conv2.bias.data = bias
            conv_idx += 1
            
            residual_idx += 1
            
        elif isinstance(module, Skip):
            # Skip connection conv
            kernel, bias = get_conv_weights(conv_idx)
            if kernel is not None:
                module.conv.weight.data = kernel
                module.conv.bias.data = bias
                print(f"  skip[{skip_idx}].conv: {kernel.shape}")
            conv_idx += 1
            skip_idx += 1
    
    # Transfer final conv
    kernel, bias = get_conv_weights(conv_idx)
    if kernel is not None:
        pytorch_model.final_conv.weight.data = kernel
        pytorch_model.final_conv.bias.data = bias
        print(f"  final_conv: {kernel.shape}")
    
    print(f"\nTransferred {conv_idx+1} conv layers, {bn_idx} bn layers")
    return True


def verify_model(pytorch_model, keras_path):
    """Verify PyTorch model produces similar outputs to Keras model"""
    try:
        import tensorflow as tf
        keras_model = tf.keras.models.load_model(keras_path, compile=False)
    except Exception as e:
        print(f"Could not load Keras model for verification: {e}")
        return
    
    # Create test input
    # Input shape for SpliceAI-10k: at least 10000 + 5000 = 15000 to produce output
    test_len = 25001
    np.random.seed(42)
    test_input = np.random.rand(1, test_len, 4).astype(np.float32)  # Keras: (N, L, C)
    
    # Keras prediction
    keras_out = keras_model.predict(test_input, verbose=0)
    print(f"Keras output shape: {keras_out.shape}")
    
    # PyTorch prediction
    pytorch_model.eval()
    with torch.no_grad():
        # PyTorch: (N, C, L)
        pt_input = torch.from_numpy(test_input.transpose(0, 2, 1))
        pt_out = pytorch_model(pt_input)
        # Back to (N, L, C) for comparison
        pt_out = pt_out.permute(0, 2, 1).numpy()
    
    print(f"PyTorch output shape: {pt_out.shape}")
    
    # Compare
    if keras_out.shape == pt_out.shape:
        max_diff = np.max(np.abs(keras_out - pt_out))
        mean_diff = np.mean(np.abs(keras_out - pt_out))
        print(f"Max difference: {max_diff:.6f}")
        print(f"Mean difference: {mean_diff:.6f}")
        
        if max_diff < 0.01:
            print("✓ Conversion successful - outputs match!")
            return True
        else:
            print("⚠ Outputs differ significantly - check weight mapping")
            return False
    else:
        print(f"⚠ Output shapes don't match: Keras {keras_out.shape} vs PyTorch {pt_out.shape}")
        return False


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Find Keras models
    keras_files = []
    for ext in ['.keras', '.h5']:
        for f in os.listdir(script_dir):
            if f.startswith('spliceai') and f.endswith(ext) and not f.endswith('.pt'):
                keras_files.append(os.path.join(script_dir, f))
    
    # Remove duplicates (prefer .keras over .h5)
    seen_models = {}
    for f in keras_files:
        base = os.path.splitext(os.path.basename(f))[0]
        if base not in seen_models or f.endswith('.keras'):
            seen_models[base] = f
    
    keras_files = list(seen_models.values())
    print(f"Found {len(keras_files)} Keras models to convert:")
    for f in keras_files:
        print(f"  {f}")
    
    if not keras_files:
        print("No Keras models found!")
        return
    
    # Convert each model
    for keras_path in sorted(keras_files):
        print(f"\n{'='*60}")
        print(f"Converting: {keras_path}")
        print('='*60)
        
        # Create PyTorch model
        pytorch_model = create_spliceai_model(flanking_size=10000)
        
        # Convert weights
        success = convert_keras_weights(keras_path, pytorch_model)
        
        if success:
            # Save PyTorch model
            base = os.path.splitext(os.path.basename(keras_path))[0]
            pt_path = os.path.join(script_dir, f"{base}.pt")
            torch.save(pytorch_model.state_dict(), pt_path)
            print(f"\n✓ Saved: {pt_path}")
            
            # Verify
            print("\nVerifying conversion...")
            verify_model(pytorch_model, keras_path)
        else:
            print(f"\n✗ Failed to convert {keras_path}")
    
    print(f"\n{'='*60}")
    print("Conversion complete!")
    print("You can now use the PyTorch models with the updated SpliceAI code.")


if __name__ == '__main__':
    main()
