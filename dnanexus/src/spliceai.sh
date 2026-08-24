#!/bin/bash

set -e  # Exit on error

main() {

    # download input files and get number of SNPs asked to condition on
    echo "########### Downloading files to worker ###########"
    export PATH="/home/dnanexus/.local/bin:${PATH}"

    # ========================================
    # Install SpliceAI
    # ========================================
    echo "########### Installing SpliceAI ###########"
    cd SpliceAI_printEVERYTHING
    pip3 install -e . --user
    cd ../

    # ========================================
    # Configure CUDA environment
    # ========================================
    echo "########### Configuring CUDA environment ###########"

    # Set CUDA paths (in case they're not automatically set)
    export CUDA_HOME=/usr/local/cuda
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    export PATH=/usr/local/cuda/bin:$PATH

    # Print CUDA configuration
    if [ -d "$CUDA_HOME" ]; then
        echo "CUDA_HOME: $CUDA_HOME"
        ls -la $CUDA_HOME/lib64/libcudart.so* 2>/dev/null || echo "libcudart.so not found in expected location"
    fi

    # Verify TensorFlow can see GPU
    echo "########### Verifying TensorFlow GPU access ###########"
    python3 -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print(f'TensorFlow found {len(gpus)} GPU(s): {gpus}'); exit(0 if len(gpus) > 0 else 1)" || {
        echo "ERROR: TensorFlow cannot see any GPUs!"
        echo "Driver version:"
        nvidia-smi --query-gpu=driver_version --format=csv,noheader
        echo "CUDA runtime version expected by TensorFlow:"
        python3 -c "import tensorflow as tf; print(tf.sysconfig.get_build_info())" || true
        echo ""
        echo "This error usually means:"
        echo "  1. CUDA driver version is too old for TensorFlow's CUDA runtime"
        echo "  2. CUDA libraries are not in LD_LIBRARY_PATH"
        echo "  3. GPU instance type is not properly configured"
        exit 1
    }

    ## download input files
    echo "Download inputs"
    dx download "file-GkZ575jJVJzjpYKfvg210FZV"
    dx download "file-GkZ575jJVJzpG16vFPBzg6xV"
    dx download "file-J4VPBp8JP7JQvPBPx42ZzB4Q"

    for i in ${file_prefixes}; do
        dx download "project-J252BXQJG5FZ6z62xq39FJ3p:/spliceai/input/${i}.bcf"
        dx download "project-J252BXQJG5FZ6z62xq39FJ3p:/spliceai/input/${i}.bcf.csi"

        spliceai -I ${i}.bcf -O annot-${i}.vcf.gz -D 500 -M 1  -R GCA_000001405.15_GRCh38_no_alt_analysis_set.fna -A MANE.GRCh38.v1.4.ensembl_genomic.txt -B ${B} -T ${T}
        dx upload annot-${i}.vcf.gz --path "project-J252BXQJG5FZ6z62xq39FJ3p:/spliceai/output/"
        rm annot-${i}.vcf.gz
        rm ${i}.bcf
    done

    #### Nonsense output
    dx-jobutil-add-output output "$output" --class=string
}
