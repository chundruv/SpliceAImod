# SpliceAI PyTorch Model Definition
# Based on the original Keras architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualUnit(nn.Module):
    """Residual unit with dilated convolutions"""
    
    def __init__(self, l, w, ar):
        """
        Args:
            l: Number of channels
            w: Kernel width
            ar: Atrous rate (dilation)
        """
        super().__init__()
        self.batchnorm1 = nn.BatchNorm1d(l)
        self.batchnorm2 = nn.BatchNorm1d(l)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        
        # Calculate padding to maintain sequence length
        # For Conv1d with dilation: padding = (w-1) * ar // 2 for same output size
        padding = (w - 1) * ar // 2
        self.conv1 = nn.Conv1d(l, l, w, dilation=ar, padding=padding)
        self.conv2 = nn.Conv1d(l, l, w, dilation=ar, padding=padding)

    def forward(self, x, skip):
        out = self.conv1(self.relu1(self.batchnorm1(x)))
        out = self.conv2(self.relu2(self.batchnorm2(out)))
        return x + out, skip


class Cropping1D(nn.Module):
    """Crop 1D sequence from both ends"""
    
    def __init__(self, cropping):
        """
        Args:
            cropping: Tuple (left_crop, right_crop)
        """
        super().__init__()
        self.cropping = cropping

    def forward(self, x):
        if self.cropping[1] > 0:
            return x[:, :, self.cropping[0]:-self.cropping[1]]
        else:
            return x[:, :, self.cropping[0]:]


class Skip(nn.Module):
    """Skip connection with 1x1 convolution"""
    
    def __init__(self, l):
        super().__init__()
        self.conv = nn.Conv1d(l, l, 1)

    def forward(self, x, skip):
        return x, self.conv(x) + skip


class SpliceAI(nn.Module):
    """
    SpliceAI model architecture in PyTorch
    
    Original architecture from Jaganathan et al. 2019:
    - Initial 1x1 convolution from 4 channels to L channels
    - Stack of residual units with dilated convolutions
    - Skip connections every 4 residual units
    - Final 1x1 convolution to 3 output channels
    - Cropping and softmax
    """
    
    def __init__(self, L=32, W=None, AR=None):
        """
        Args:
            L: Number of convolution filters (default 32)
            W: Array of kernel widths for each residual unit
            AR: Array of atrous rates (dilations) for each residual unit
        """
        super(SpliceAI, self).__init__()
        
        # Default to 10000bp flanking size parameters
        if W is None:
            W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
                           21, 21, 21, 21, 41, 41, 41, 41])
        if AR is None:
            AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
                            10, 10, 10, 10, 25, 25, 25, 25])
        
        self.L = L
        self.W = W
        self.AR = AR
        
        # Calculate context length (receptive field)
        self.CL = 2 * np.sum(AR * (W - 1))
        
        # Initial convolution: 4 input channels (one-hot encoded ACGT) -> L channels
        self.initial_conv = nn.Conv1d(4, L, 1)
        self.initial_skip = Skip(L)
        
        # Build residual units with skip connections
        self.residual_units = nn.ModuleList()
        for i, (w, r) in enumerate(zip(W, AR)):
            self.residual_units.append(ResidualUnit(L, w, r))
            if (i + 1) % 4 == 0:
                self.residual_units.append(Skip(L))
        
        # Final convolution: L channels -> 3 output channels
        # (Neither, Acceptor, Donor)
        self.final_conv = nn.Conv1d(L, 3, 1)
        
        # Store minimum required input length
        self.min_input_length = self.CL + 1

    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch, 4, sequence_length)
               One-hot encoded DNA sequence (ACGT)
        
        Returns:
            Output tensor of shape (batch, 3, output_length)
            Softmax probabilities for (Neither, Acceptor, Donor) at each position
        """
        # Validate input
        if x.dim() != 3:
            raise ValueError(f"SpliceAI expects 3D input (N, C, L), got {x.dim()}D with shape {x.shape}")
        
        batch_size, channels, seq_len = x.shape
        
        if channels != 4:
            raise ValueError(f"SpliceAI expects 4 channels (one-hot encoded), got {channels}")
        
        if seq_len < self.min_input_length:
            raise ValueError(
                f"SpliceAI-{self.CL//2} requires input sequence length >= {self.min_input_length}, "
                f"got {seq_len}. For SpliceAI-10k, use at least 10001 (ideally 25001)."
            )
        
        # Initial convolution
        x = self.initial_conv(x)
        x, skip = self.initial_skip(x, 0)
        
        # Residual blocks with skip connections
        for m in self.residual_units:
            x, skip = m(x, skip)
        
        # Crop to remove edge effects (CL // 2 from each end)
        crop_amount = self.CL // 2
        if crop_amount > 0 and skip.size(2) > 2 * crop_amount:
            skip = skip[:, :, crop_amount:-crop_amount]
        elif crop_amount > 0:
            raise ValueError(
                f"Cannot crop {crop_amount} from each end of sequence with length {skip.size(2)}. "
                f"Input sequence was too short."
            )
        
        # Final convolution and softmax
        out = self.final_conv(skip)
        return F.softmax(out, dim=1)

    @staticmethod
    def from_keras_weights(keras_model_path, device='cpu'):
        """
        Load weights from a Keras .h5 or .keras model file
        
        Args:
            keras_model_path: Path to Keras model file
            device: Target device for the model
        
        Returns:
            SpliceAI model with loaded weights
        """
        import h5py
        
        # Create PyTorch model with default parameters
        model = SpliceAI()
        
        # Load Keras weights
        with h5py.File(keras_model_path, 'r') as f:
            # Navigate the Keras model structure
            # This will need to map Keras layer names to PyTorch layer names
            _load_keras_weights_to_pytorch(f, model)
        
        model = model.to(device)
        model.eval()
        return model


def _load_keras_weights_to_pytorch(h5_file, pytorch_model):
    """
    Map Keras weights to PyTorch model
    
    Keras Conv1D: (kernel_size, in_channels, out_channels)
    PyTorch Conv1d: (out_channels, in_channels, kernel_size)
    
    Keras BatchNorm: [gamma, beta, mean, variance]
    PyTorch BatchNorm: weight, bias, running_mean, running_var
    """
    import re
    
    def get_weights(h5_file, layer_name):
        """Extract weights from Keras HDF5 file"""
        try:
            if 'model_weights' in h5_file:
                base = h5_file['model_weights']
            else:
                base = h5_file
            
            if layer_name in base:
                layer = base[layer_name]
                if layer_name in layer:
                    layer = layer[layer_name]
                
                weights = []
                for key in sorted(layer.keys()):
                    if 'kernel' in key or 'gamma' in key or 'beta' in key or 'mean' in key or 'variance' in key:
                        weights.append(np.array(layer[key]))
                return weights
        except Exception as e:
            print(f"Warning: Could not load weights for {layer_name}: {e}")
        return None
    
    # Map initial conv
    weights = get_weights(h5_file, 'conv1d')
    if weights:
        # Keras: (kernel_size, in_channels, out_channels)
        # PyTorch: (out_channels, in_channels, kernel_size)
        kernel = weights[0]
        kernel = np.transpose(kernel, (2, 1, 0))
        pytorch_model.initial_conv.weight.data = torch.from_numpy(kernel.copy())
        if len(weights) > 1:
            pytorch_model.initial_conv.bias.data = torch.from_numpy(weights[1].copy())
    
    # Map skip and residual units
    # This is complex due to Keras naming conventions
    # We need to iterate through the layers and match them
    
    # For now, log a warning that weight loading is not fully implemented
    print("Warning: Full Keras weight loading not implemented. Please use pre-converted PyTorch models.")


def create_spliceai_model(flanking_size=10000, device='cpu', precision='fp32', compile_model=False):
    """
    Create a SpliceAI model with appropriate hyperparameters
    
    Args:
        flanking_size: Flanking sequence size (80, 400, 2000, or 10000)
        device: Target device
        precision: Precision mode ('fp32', 'fp16', 'bf16')
        compile_model: Whether to use torch.compile()
    
    Returns:
        Configured SpliceAI model
    """
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
    model = model.to(device)
    model.eval()
    
    # Apply precision settings
    if precision == 'fp16':
        model = model.half()
    elif precision == 'bf16':
        model = model.to(torch.bfloat16)
    
    # Compile model for optimized inference
    if compile_model and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode='reduce-overhead')
        except Exception as e:
            print(f"Warning: torch.compile() failed: {e}")
    
    return model


def load_pytorch_model(model_path, device='cpu', precision='fp32', compile_model=False):
    """
    Load a PyTorch SpliceAI model from file
    
    Args:
        model_path: Path to .pt or .pth file
        device: Target device
        precision: Precision mode
        compile_model: Whether to use torch.compile()
    
    Returns:
        Loaded model
    """
    # Accept three on-disk forms: a training checkpoint wrapping a state dict,
    # a bare state dict, or a pickled nn.Module.
    #
    # The bare-state-dict test used to be
    #     not any(k.startswith('initial_conv') for k in checkpoint)
    # which is inverted: EVERY SpliceAI state dict begins with initial_conv.*,
    # so all five shipped spliceai{1..5}.pt fell through to the nn.Module branch
    # and load_pytorch_model() returned the OrderedDict itself, failing at the
    # next line with "'collections.OrderedDict' object has no attribute 'to'".
    # Dispatch on the object's type instead of on a parameter name.
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, nn.Module):
        model = checkpoint
    elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model = create_spliceai_model(device=device, precision='fp32')
        model.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict):
        model = create_spliceai_model(device=device, precision='fp32')
        model.load_state_dict(checkpoint)
    else:
        raise TypeError(
            "Unrecognised checkpoint at {}: expected an nn.Module, a dict with "
            "'model_state_dict', or a bare state dict; got {}".format(
                model_path, type(checkpoint).__name__))


    model = model.to(device)
    model.eval()
    
    # Apply precision
    if precision == 'fp16':
        model = model.half()
    elif precision == 'bf16':
        model = model.to(torch.bfloat16)

    # Compile
    if compile_model and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode='reduce-overhead')
        except Exception as e:
            print(f"Warning: torch.compile() failed: {e}")
    
    return model
