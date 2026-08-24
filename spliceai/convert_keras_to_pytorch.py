#!/usr/bin/env python3
"""
Keras to PyTorch SpliceAI Model Converter

This script converts SpliceAI Keras models (.h5 or .keras) to PyTorch format (.pt).

Usage:
    python convert_keras_to_pytorch.py --input models/spliceai1.h5 --output models/spliceai1.pt
    python convert_keras_to_pytorch.py --convert-all  # Convert all 5 models
    python convert_keras_to_pytorch.py --validate models/spliceai1.pt  # Validate conversion
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import h5py
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# Import the PyTorch model architecture
from spliceai.models.pytorch_model import SpliceAI, create_spliceai_model


def load_keras_model_structure(h5_path):
    """
    Analyze the Keras model structure to understand layer naming.
    
    Returns dict with layer info.
    """
    with h5py.File(h5_path, 'r') as f:
        def print_structure(name, obj):
            logger.debug(f"  {name}: {type(obj)}")
        
        logger.info(f"Analyzing model structure: {h5_path}")
        f.visititems(print_structure)
        
        # Find the weights group
        if 'model_weights' in f:
            weights_group = f['model_weights']
            logger.info("Found 'model_weights' group")
        else:
            weights_group = f
            logger.info("Using root as weights group")
        
        layer_names = list(weights_group.keys())
        logger.info(f"Found {len(layer_names)} layer groups")
        
        return layer_names


def get_keras_layer_weights(h5_file, layer_name):
    """
    Extract weights from a Keras layer in the HDF5 file.
    
    Keras stores weights in nested groups:
    model_weights/layer_name/layer_name/kernel:0
    model_weights/layer_name/layer_name/bias:0
    """
    weights = {}
    
    # Try different possible paths
    possible_paths = [
        f'model_weights/{layer_name}/{layer_name}',
        f'model_weights/{layer_name}',
        f'{layer_name}/{layer_name}',
        f'{layer_name}',
    ]
    
    for path in possible_paths:
        try:
            if path in h5_file:
                layer_group = h5_file[path]
                for key in layer_group.keys():
                    data = np.array(layer_group[key])
                    # Clean up key name (remove :0 suffix)
                    clean_key = key.replace(':0', '')
                    weights[clean_key] = data
                    logger.debug(f"  Loaded {layer_name}/{clean_key}: shape {data.shape}")
                if weights:
                    return weights
        except Exception as e:
            continue
    
    return weights


def convert_conv_weights(keras_weights):
    """
    Convert Keras Conv1D weights to PyTorch Conv1d format.
    
    Keras Conv1D: (kernel_size, in_channels, out_channels)
    PyTorch Conv1d: (out_channels, in_channels, kernel_size)
    """
    if 'kernel' not in keras_weights:
        return None, None
    
    kernel = keras_weights['kernel']
    # Transpose: (kernel_size, in_channels, out_channels) -> (out_channels, in_channels, kernel_size)
    kernel = np.transpose(kernel, (2, 1, 0))
    
    bias = keras_weights.get('bias', None)
    
    return kernel, bias


def convert_batchnorm_weights(keras_weights):
    """
    Convert Keras BatchNormalization weights to PyTorch BatchNorm1d format.
    
    Keras BatchNorm: gamma (weight), beta (bias), moving_mean, moving_variance
    PyTorch BatchNorm1d: weight, bias, running_mean, running_var
    """
    result = {}
    
    if 'gamma' in keras_weights:
        result['weight'] = keras_weights['gamma']
    if 'beta' in keras_weights:
        result['bias'] = keras_weights['beta']
    if 'moving_mean' in keras_weights:
        result['running_mean'] = keras_weights['moving_mean']
    if 'moving_variance' in keras_weights:
        result['running_var'] = keras_weights['moving_variance']
    
    return result


def analyze_keras_model(h5_path):
    """
    Analyze Keras model and return layer structure.
    """
    layer_info = []
    
    with h5py.File(h5_path, 'r') as f:
        # Get layer names
        if 'model_weights' in f:
            base = f['model_weights']
        else:
            base = f
        
        for name in base.keys():
            weights = get_keras_layer_weights(f, name)
            if weights:
                layer_info.append({
                    'name': name,
                    'weights': {k: v.shape for k, v in weights.items()}
                })
                logger.info(f"Layer {name}: {layer_info[-1]['weights']}")
    
    return layer_info


def convert_keras_to_pytorch(keras_path, pytorch_path=None, validate=True, device='cpu'):
    """
    Convert a Keras SpliceAI model to PyTorch format.
    
    Args:
        keras_path: Path to Keras .h5 or .keras file
        pytorch_path: Output path for PyTorch .pt file (auto-generated if None)
        validate: Whether to validate the conversion
        device: Device for validation
    
    Returns:
        PyTorch model with loaded weights
    """
    logger.info(f"Converting {keras_path} to PyTorch format")
    
    # Create PyTorch model
    pytorch_model = create_spliceai_model(flanking_size=10000, device='cpu')
    
    # Load Keras weights
    with h5py.File(keras_path, 'r') as f:
        # Analyze structure first
        if 'model_weights' in f:
            base = f['model_weights']
        else:
            base = f
        
        layer_names = list(base.keys())
        logger.info(f"Found {len(layer_names)} layers in Keras model")
        
        # =======================
        # Map initial convolution
        # =======================
        # Keras layer name pattern: conv1d, conv1d_1, etc. or conv_0, conv_1
        initial_conv_loaded = False
        for name in layer_names:
            if name.startswith('conv') and not initial_conv_loaded:
                weights = get_keras_layer_weights(f, name)
                if weights and 'kernel' in weights:
                    kernel, bias = convert_conv_weights(weights)
                    if kernel.shape == (32, 4, 1):  # L=32, in=4, kernel=1
                        pytorch_model.initial_conv.weight.data = torch.from_numpy(kernel.copy()).float()
                        if bias is not None:
                            pytorch_model.initial_conv.bias.data = torch.from_numpy(bias.copy()).float()
                        logger.info(f"Loaded initial_conv from {name}")
                        initial_conv_loaded = True
        
        # =======================
        # Map Skip connections
        # =======================
        # In Keras, these are typically named conv1d_N where N follows a pattern
        # Initial skip: after initial conv
        # Then every 4 residual units: at indices 4, 8, 12, 16 (0-indexed: 3, 7, 11, 15)
        
        # Find all conv1d layers (skip connections use 1x1 convs)
        conv_layers = [n for n in layer_names if n.startswith('conv')]
        conv_layers_sorted = sorted(conv_layers, key=lambda x: int(x.split('_')[-1]) if '_' in x and x.split('_')[-1].isdigit() else 0)
        
        logger.debug(f"Conv layers (sorted): {conv_layers_sorted}")
        
        # =======================
        # Map Residual Units
        # =======================
        # Each residual unit has:
        # - 2 BatchNorm layers
        # - 2 Conv layers
        
        # Find batch norm layers
        bn_layers = [n for n in layer_names if 'batch_normalization' in n.lower() or 'bn' in n.lower()]
        bn_layers_sorted = sorted(bn_layers, key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else 0)
        
        logger.info(f"Found {len(bn_layers_sorted)} batch norm layers")
        logger.debug(f"BatchNorm layers: {bn_layers_sorted}")
        
        # Build mapping of Keras layers to PyTorch modules
        # The PyTorch model structure:
        # - initial_conv
        # - initial_skip (Skip with 1x1 conv)
        # - residual_units: [ResidualUnit, ResidualUnit, ResidualUnit, ResidualUnit, Skip, 
        #                    ResidualUnit, ResidualUnit, ResidualUnit, ResidualUnit, Skip, ...]
        
        # Count residual units and skip connections in PyTorch model
        res_unit_idx = 0
        skip_idx = 0
        bn_idx = 0
        conv_idx = 1  # Start at 1 since 0 is initial_conv
        
        # Load initial skip
        for name in layer_names:
            weights = get_keras_layer_weights(f, name)
            if weights and 'kernel' in weights:
                kernel, bias = convert_conv_weights(weights)
                # Initial skip is 1x1 conv with L->L (32->32)
                if kernel.shape == (32, 32, 1) and skip_idx == 0:
                    pytorch_model.initial_skip.conv.weight.data = torch.from_numpy(kernel.copy()).float()
                    if bias is not None:
                        pytorch_model.initial_skip.conv.bias.data = torch.from_numpy(bias.copy()).float()
                    logger.info(f"Loaded initial_skip from conv layer")
                    skip_idx = 1
                    break
        
        # Now load residual units
        # The model has 16 residual units with skip every 4
        pytorch_module_idx = 0
        keras_bn_idx = 0
        keras_conv_idx = 2  # Start after initial_conv and initial_skip
        
        for pytorch_module_idx, module in enumerate(pytorch_model.residual_units):
            if isinstance(module, nn.Module) and module.__class__.__name__ == 'ResidualUnit':
                # Load 2 batch norms
                for bn_num, bn_layer in enumerate([module.batchnorm1, module.batchnorm2]):
                    if keras_bn_idx < len(bn_layers_sorted):
                        bn_name = bn_layers_sorted[keras_bn_idx]
                        weights = get_keras_layer_weights(f, bn_name)
                        if weights:
                            bn_weights = convert_batchnorm_weights(weights)
                            if 'weight' in bn_weights:
                                bn_layer.weight.data = torch.from_numpy(bn_weights['weight'].copy()).float()
                            if 'bias' in bn_weights:
                                bn_layer.bias.data = torch.from_numpy(bn_weights['bias'].copy()).float()
                            if 'running_mean' in bn_weights:
                                bn_layer.running_mean.data = torch.from_numpy(bn_weights['running_mean'].copy()).float()
                            if 'running_var' in bn_weights:
                                bn_layer.running_var.data = torch.from_numpy(bn_weights['running_var'].copy()).float()
                            logger.debug(f"Loaded BatchNorm from {bn_name}")
                        keras_bn_idx += 1
                
                # Load 2 convs
                for conv_num, conv_layer in enumerate([module.conv1, module.conv2]):
                    if keras_conv_idx < len(conv_layers_sorted):
                        conv_name = conv_layers_sorted[keras_conv_idx]
                        weights = get_keras_layer_weights(f, conv_name)
                        if weights and 'kernel' in weights:
                            kernel, bias = convert_conv_weights(weights)
                            # Check shape matches
                            if kernel.shape[0] == conv_layer.weight.shape[0]:
                                conv_layer.weight.data = torch.from_numpy(kernel.copy()).float()
                                if bias is not None and conv_layer.bias is not None:
                                    conv_layer.bias.data = torch.from_numpy(bias.copy()).float()
                                logger.debug(f"Loaded Conv from {conv_name}")
                        keras_conv_idx += 1
                        
            elif hasattr(module, 'conv'):  # Skip module
                # Load skip connection conv
                if keras_conv_idx < len(conv_layers_sorted):
                    conv_name = conv_layers_sorted[keras_conv_idx]
                    weights = get_keras_layer_weights(f, conv_name)
                    if weights and 'kernel' in weights:
                        kernel, bias = convert_conv_weights(weights)
                        if kernel.shape == (32, 32, 1):
                            module.conv.weight.data = torch.from_numpy(kernel.copy()).float()
                            if bias is not None:
                                module.conv.bias.data = torch.from_numpy(bias.copy()).float()
                            logger.debug(f"Loaded Skip conv from {conv_name}")
                    keras_conv_idx += 1
        
        # =======================
        # Map final convolution
        # =======================
        # Final conv is L->3 (32->3) with kernel size 1
        for name in reversed(conv_layers_sorted):
            weights = get_keras_layer_weights(f, name)
            if weights and 'kernel' in weights:
                kernel, bias = convert_conv_weights(weights)
                if kernel.shape == (3, 32, 1):  # out=3, in=32, kernel=1
                    pytorch_model.final_conv.weight.data = torch.from_numpy(kernel.copy()).float()
                    if bias is not None:
                        pytorch_model.final_conv.bias.data = torch.from_numpy(bias.copy()).float()
                    logger.info(f"Loaded final_conv from {name}")
                    break
    
    pytorch_model.eval()
    
    # =======================
    # Validation
    # =======================
    if validate:
        logger.info("Validating conversion...")
        validation_passed = validate_conversion(keras_path, pytorch_model, device)
        if validation_passed:
            logger.info("✓ Validation PASSED: PyTorch model outputs match Keras model")
        else:
            logger.warning("✗ Validation FAILED: Outputs differ significantly")
    
    # =======================
    # Save PyTorch model
    # =======================
    if pytorch_path is None:
        pytorch_path = str(keras_path).replace('.h5', '.pt').replace('.keras', '.pt')
    
    torch.save(pytorch_model.state_dict(), pytorch_path)
    logger.info(f"Saved PyTorch model to {pytorch_path}")
    
    return pytorch_model


def validate_conversion(keras_path, pytorch_model, device='cpu'):
    """
    Validate PyTorch conversion by comparing outputs with Keras.
    
    Returns True if outputs match within tolerance.
    """
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError:
        logger.warning("TensorFlow not available for validation")
        return None
    
    # Load Keras model
    keras_model = keras.models.load_model(keras_path, compile=False)
    
    # Generate random test input
    np.random.seed(42)
    # Input shape for Keras: (batch, sequence, channels) = (1, 15001, 4)
    # This is for 10000bp flanking with context
    seq_len = 15001  # 10000 + 5001 for context
    test_input = np.random.rand(1, seq_len, 4).astype(np.float32)
    
    # Get Keras prediction
    keras_output = keras_model.predict(test_input, verbose=0)
    
    # Get PyTorch prediction
    pytorch_model = pytorch_model.to(device)
    pytorch_model.eval()
    
    # Convert input: (batch, seq, channels) -> (batch, channels, seq)
    pytorch_input = torch.from_numpy(np.transpose(test_input, (0, 2, 1))).float().to(device)
    
    with torch.no_grad():
        pytorch_output = pytorch_model(pytorch_input)
        # Convert output: (batch, channels, seq) -> (batch, seq, channels)
        pytorch_output = pytorch_output.cpu().numpy().transpose(0, 2, 1)
    
    # Compare outputs
    # Keras output shape: (batch, output_seq, 3)
    # PyTorch output should be the same after transpose
    
    max_diff = np.max(np.abs(keras_output - pytorch_output))
    mean_diff = np.mean(np.abs(keras_output - pytorch_output))
    
    logger.info(f"Output shapes - Keras: {keras_output.shape}, PyTorch: {pytorch_output.shape}")
    logger.info(f"Max absolute difference: {max_diff:.6f}")
    logger.info(f"Mean absolute difference: {mean_diff:.6f}")
    
    # Check correlation
    keras_flat = keras_output.flatten()
    pytorch_flat = pytorch_output.flatten()
    correlation = np.corrcoef(keras_flat, pytorch_flat)[0, 1]
    logger.info(f"Correlation coefficient: {correlation:.6f}")
    
    # Validation passes if max diff < 1e-4 and correlation > 0.999
    return max_diff < 1e-4 and correlation > 0.999


def convert_all_models(models_dir, validate=True):
    """
    Convert all SpliceAI Keras models in a directory to PyTorch format.
    """
    models_dir = Path(models_dir)
    
    # Find all Keras models
    h5_models = list(models_dir.glob('spliceai*.h5'))
    keras_models = list(models_dir.glob('spliceai*.keras'))
    
    all_models = h5_models + keras_models
    
    if not all_models:
        logger.error(f"No Keras models found in {models_dir}")
        return
    
    logger.info(f"Found {len(all_models)} Keras models to convert")
    
    for model_path in sorted(all_models):
        # Skip if .pt already exists
        pt_path = model_path.with_suffix('.pt')
        if pt_path.exists():
            logger.info(f"Skipping {model_path.name} - {pt_path.name} already exists")
            continue
        
        try:
            convert_keras_to_pytorch(str(model_path), str(pt_path), validate=validate)
        except Exception as e:
            logger.error(f"Failed to convert {model_path}: {e}")
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='Convert Keras SpliceAI models to PyTorch')
    parser.add_argument('--input', '-i', help='Input Keras model path (.h5 or .keras)')
    parser.add_argument('--output', '-o', help='Output PyTorch model path (.pt)')
    parser.add_argument('--convert-all', action='store_true', help='Convert all models in models/ directory')
    parser.add_argument('--models-dir', default='models', help='Directory containing Keras models')
    parser.add_argument('--validate', action='store_true', default=True, help='Validate conversion')
    parser.add_argument('--no-validate', action='store_false', dest='validate', help='Skip validation')
    parser.add_argument('--analyze', action='store_true', help='Analyze Keras model structure only')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.analyze and args.input:
        analyze_keras_model(args.input)
        return
    
    if args.convert_all:
        # Find models directory relative to this script
        script_dir = Path(__file__).parent
        models_dir = script_dir / args.models_dir
        if not models_dir.exists():
            models_dir = Path(args.models_dir)
        convert_all_models(models_dir, validate=args.validate)
    elif args.input:
        convert_keras_to_pytorch(args.input, args.output, validate=args.validate)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
