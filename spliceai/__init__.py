# Original source code modified to add prediction batching support by Invitae in 2021.
# Modifications copyright (c) 2021 Invitae Corporation.

# Converted to PyTorch with FP16/BF16 support

import signal
from importlib.metadata import version

try:
    signal.signal(signal.SIGINT, lambda x, y: exit(0))
except ValueError:
    # Continue if we're not able to set the signal handler due to which thread is running the code
    pass

__version__ = version('spliceai')
