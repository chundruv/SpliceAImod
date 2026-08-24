SHELL=/bin/bash -e -x -o pipefail
all:
	# Temporarily move DNAnexus apt config to allow package installation
	sudo mv /etc/apt/apt.conf.d/99dnanexus /tmp/

	# Add NVIDIA CUDA repository for Ubuntu 24.04
	wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
	sudo dpkg -i cuda-keyring_1.1-1_all.deb
	sudo apt-get update -qq

	# Install CUDA 12.6 toolkit and forward compatibility packages
	# Forward compatibility allows driver 470+ to work with CUDA 12.x
	sudo DEBIAN_FRONTEND=noninteractive apt-get -y install cuda-toolkit-12-6 cuda-compat-12-6

	# Configure ld.so to find CUDA forward compatibility libraries
	sudo bash -c "echo /usr/local/cuda/compat > /etc/ld.so.conf.d/nvidia-compat.conf"
	sudo bash -c "echo /usr/local/cuda-12.6/lib64 >> /etc/ld.so.conf.d/nvidia-compat.conf"
	sudo ldconfig

	# Install Python 3.12 and development tools (default in Ubuntu 24.04)
	sudo DEBIAN_FRONTEND=noninteractive apt-get -y install \
		python3.12 \
		python3.12-dev \
		python3-pip \
		python3.12-venv \
		build-essential \
		git \
		wget \
		ca-certificates \
		pkg-config

	# Upgrade pip, setuptools, wheel (ignore system-installed versions)
	python3 -m pip install --upgrade --ignore-installed --break-system-packages pip setuptools wheel

	# Install TensorFlow 2.20 with CUDA support (Python 3.12 + CUDA 12.6)
	# This will automatically install compatible keras and numpy versions
	python3 -m pip install --ignore-installed --break-system-packages tensorflow[and-cuda]==2.20.0

	# Install SpliceAI dependencies (let TensorFlow's versions of keras/numpy take precedence)
	python3 -m pip install --break-system-packages \
		pandas==2.2.3 \
		pysam==0.22.1 \
		pyfaidx==0.8.1.3 \
		intervaltree==3.1.0

	# Verify TensorFlow installation
	python3 -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('CUDA build:', tf.test.is_built_with_cuda())"

	# Clean up
	sudo rm -f cuda-keyring_1.1-1_all.deb

	# Restore DNAnexus apt config
	sudo mv /tmp/99dnanexus /etc/apt/apt.conf.d/99dnanexus

	# Print final status
	@echo "========================================"
	@echo "Asset Build Complete!"
	@echo "========================================"
	@echo "CUDA Toolkit: 12.6"
	@echo "TensorFlow: 2.20.0 (with CUDA 12.6 forward compatibility)"
	@echo "Python: $$(python3 --version)"
	@echo "Ubuntu: 24.04 LTS"
	@echo "========================================"
