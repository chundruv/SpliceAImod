## SpliceAI: A deep learning-based tool to identify splice variants
[![release](https://img.shields.io/badge/release-v1.3.1-orange.svg)](https://img.shields.io/badge/release-v1.3.1-orange.svg)
[![license](https://img.shields.io/badge/license-GPLv3-green.svg)](https://img.shields.io/badge/license-GPLv3-green.svg)
[![downloads](https://pepy.tech/badge/spliceai)](https://pepy.tech/badge/spliceai)

This package annotates genetic variants with their predicted effect on splicing, as described in [Jaganathan *et al*, Cell 2019 in press](https://doi.org/10.1016/j.cell.2018.12.015). The annotations for all possible substitutions, 1 base insertions, and 1-4 base deletions within genes are available [here](https://basespace.illumina.com/s/otSPW8hnhaZR) for download. These annotations are free for academic and not-for-profit use; other use requires a commercial license from Illumina, Inc.

### License
SpliceAI source code is provided under the [GPLv3 license](LICENSE). SpliceAI includes several third party packages provided under other open source licenses, please see [NOTICE](NOTICE) for additional details. The trained models used by SpliceAI (located in this package at spliceai/models) are provided under the [CC BY NC 4.0](LICENSE) license for academic and non-commercial use; other use requires a commercial license from Illumina, Inc.

### Updates - Kartik Chundru - December 2025

This fork is a version of SpliceAI adapted (from the amazing work done by Geert Vandeweyer, who built upon the work by the Invitae team who modified the original Illumina code) to work with PyTorch using fp16 precision (with optional bf16 support), and optimized for multi-GPU setups with batching support to make the model inference as fast as possible. In addition, the code has been updated to output more information per variant in the VCF output.

More recent annotation files were also added, including GENCODE v49 and MANE v1.4. GTF files for these were converted to the SpliceAI annotation format using https://github.com/bw2/annotation-utils.

Changes made:
- Converted models to PyTorch
- Added fp16/bf16 support for modern GPUs
- Added multi-GPU support
- Added batching support
- Updated code to now mask all annotated splice sites in the given annotation file, not just the nearest one which was the case in the original implementation
- Updated models to pytorch format
- Updated output to include more information per variant
- Output extended inference

### Installation

```
pip install torch pysam pyfaidx pandas numpy intervaltree numba h5py psutil

[For much faster tar creation/extraction (recommended):]
sudo apt install pigz

git clone https://github.com/chundruv/SpliceAImod.git
cd SpliceAImod
python setup.py install
```

### Usage
SpliceAI can be run from the command line:
```
spliceai \
    -I input.bcf \
    -O output.tsv.gz \
    -D 500 \
    -R genome.fa \
    -A spliceai/annotations/gencode.v49.annotation.txt \
    -B 1024
```

The default output is a tsv file with the following columns:

|    Column    |    Name    |    Description    |
|------------|------------|-----------------|
| 1  | `#CHROM` | Chromosome of the variant |
| 2  | `POS` | 1-based position of the variant |
| 3  | `REF` | Reference allele |
| 4  | `ALT` | All alternate alleles of the input record (comma-separated) |
| 5  | `ALLELE` | The single alternate allele scored in this row |
| 6  | `SYMBOL` | Gene symbol of the transcript scored in this row |
| 7  | `DSM_AG` | Masked delta score, acceptor gain |
| 8  | `DSM_AL` | Masked delta score, acceptor loss |
| 9  | `DSM_DG` | Masked delta score, donor gain |
| 10 | `DSM_DL` | Masked delta score, donor loss |
| 11 | `DS_AG` | Raw delta score, acceptor gain (`P_alt - P_ref` at `DP_AG`) |
| 12 | `DS_AL` | Raw delta score, acceptor loss (`P_ref - P_alt` at `DP_AL`) |
| 13 | `DS_DG` | Raw delta score, donor gain |
| 14 | `DS_DL` | Raw delta score, donor loss |
| 15 | `DP_AG` | Position of the acceptor gain, in nt relative to the variant (negative = upstream) |
| 16 | `DP_AL` | Position of the acceptor loss, in nt relative to the variant |
| 17 | `DP_DG` | Position of the donor gain, in nt relative to the variant |
| 18 | `DP_DL` | Position of the donor loss, in nt relative to the variant |
| 19 | `RS_AG` | Raw model acceptor probability in the ALT sequence at `DP_AG` |
| 20 | `RS_AL` | Raw model acceptor probability in the ALT sequence at `DP_AL` |
| 21 | `RS_DG` | Raw model donor probability in the ALT sequence at `DP_DG` |
| 22 | `RS_DL` | Raw model donor probability in the ALT sequence at `DP_DL` |
| 23 | `MANEselectDonors` | Annotated donor sites falling inside the context window, as `(DP,RS_REF,RS_ALT)` tuples; `.` if none |
| 24 | `MANEselectAcceptors` | Annotated acceptor sites falling inside the context window, as `(DP,RS_REF,RS_ALT)` tuples; `.` if none |
| 25 | `DonorsRSgt0.5` | Every position whose REF **or** ALT donor probability exceeds 0.5, as `(DP,RS_REF,RS_ALT)` tuples; `.` if none |
| 26 | `AcceptorsRSgt0.5` | Every position whose REF **or** ALT acceptor probability exceeds 0.5, as `(DP,RS_REF,RS_ALT)` tuples; `.` if none |
| 27 | `EVENT_CLASS` | Frame consequence of the predicted event (see below) |
| 28 | `N_EXONS` | Number of exons in the annotation record for this gene |
| 29 | `PREDICTED_EVENT` | Structural event underlying the call: `no_change`, `exon_skip`, `intron_retention`, `pseudoexon`, `cryptic_shift_insertion`, `cryptic_shift_deletion`, `cryptic_extension_insertion`, `cryptic_extension_deletion`, `utr_event` or `ambiguous` |

Masked scores (`DSM_*`) apply the usual SpliceAI mask: a gain at a position that is already an annotated splice site, and a loss at a position that is not annotated, are set to 0. Unlike the original implementation, the mask considers **every** splice site of the transcript in the annotation file, not just the nearest one. The raw (`DS_*`) columns are never masked.

`EVENT_CLASS` is a composite label rather than a fixed enumeration:

- `NoChange`, `Ambiguous`, `NonCoding`, `5UTR_Event`, `3UTR_Event` — no frame call was made.
- `InFrameInsertion(Naa_p)` / `InFrameDeletion(Naa_p)` — length change is a multiple of 3; `N` is the number of residues gained/lost and `p` the fraction of the CDS affected. `InFrameInsertion_PTC(...)` marks an in-frame insertion that itself contains a stop codon.
- `Frameshift(p_Naa_Maa)_NMD` / `_NMDescape` / `_NMD_by_NewExon` / `_Ambiguous` — `p` is the fraction of the CDS upstream of the event, `N` the residues translated before it and `M` the residues to the next in-frame stop. The NMD suffix is assigned by testing whether the **event** lies more than 55 nt upstream of the last exon-exon junction; no premature stop codon is translated, so it is a positional proxy for NMD, not a prediction of decay.
- A `_Minor` suffix is appended when a nearby canonical site still scores higher than the predicted cryptic site in the ALT sequence.

With `--orig-output` the file contains only columns 1-18, in the same order.




Required parameters:
 - ```-I```: Input VCF with variants of interest.
 - ```-O```: Output VCF
 - ```-R```: Reference genome fasta file. Can be downloaded from [GRCh38/hg38](https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/001/405/GCA_000001405.15_GRCh38/seqs_for_alignment_pipelines.ucsc_ids/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna.gz).
 - ```-A```: Gene annotation file. GENCODE primary v49 (input "gencodev49", default) and MANEv1.4 (use "MANEv1.4") are provided

Optional parameters:
 - ```-D```: Maximum distance between the variant and gained/lost splice site (default: 500).
 - ```-B```: Number of predictions to collect before running models on them in batch. (default: 1 (don't batch))
 - ```-T```: PyTorch batch size for model predictions (default: auto — 64 on a single GPU, 32 per GPU when several are used, 1 or 16 on CPU depending on `--cpu-processes`). Raise it if the device has memory to spare.
 - ```-V```: Enable verbose logging during run
 - ```-t```: Specify a location to create the temporary files
 - ```-G```: Specify the GPU(s) to run on : either indexed (eg : 0,2) or 'all'. (default: 'all')
 - ```--precision```: Specify the floating point precision to use : 'fp32', 'fp16' or 'bf16' (default: 'auto', which picks the best available for your GPU)
 - ```--workers-per-gpu```: Number of workers to launch per GPU (default: 1). Will slow down performance if set too high. Can be useful in some cases, but generally not recommended.
 - ```--compile```: Use torch.compile() for optimized inference (requires PyTorch 2.0+). Recommended for best performance.
 - ```--no-cuda-graphs```: Disable CUDA Graphs optimization. Don't use for lower memory GPUs.
 - ```--batch-workers```: Number of parallel workers for batch creation/VCF reading (default: 1). Higher values can speed up batch preparation for large VCF files
 - ```--gpu-profile-interval```: GPU profiling output frequency in batches (default: 100)
 - ```--cpu-processes```: CPU proceses to use (For CPU mode)
 - ```--orig-output```: Only print original SpliceAI VCF output without additional fields (but I still print both masked and unmasked scores)
 - ```--skip-write```: Skip writing the final output and outputs tar.gz of intermediate files instead
 - ```--vcf-output```: Write to vcf instead of tsv.

NOTE: I have removed the `-M` parameter as the output now always includes both masked and raw scores.

## Precision Modes

| Mode | Description | Best For |
|------|-------------|----------|
| `fp32` | Full precision | CPU inference, debugging |
| `fp16` | Half precision | CUDA devices without bfloat16 support |
| `bf16` | BFloat16 | CUDA devices with bfloat16 support |
| `auto` | Ask the device what it supports: `fp32` on CPU, `bf16` where `torch.cuda.is_bf16_supported()`, `fp16` otherwise | Default |

**Batching Considerations:**

*Updated benchmarks:*

|    Type    |    VRAM    |    Precision    |    Batch Size    |    PyTorch Batch Size    |    Speed (predictions / hour)    |
|------------|------------|-----------------|------------------|--------------------------|----------------------------------|
| A10G       | 22GB       | bf16            | 20480            | 2048                     | ~1,100,000 pred/h                |
| V100       | 16GB       | fp16            | 10240            | 1024                     | ~1,000,000 pred/h                |
| RTX 5090   | 32GB       | bf16            | 15360            | 3072                     | ~2,900,000 pred/h                |

NOTE: These benchmarks do not include the write time to output VCF file. This can be significant for large VCF files. For large-scale predictions, we used the --skip-write option which returns a tar of the intermediate files, which can then be transfered to a CPU machine to process and output the annotated VCF so as to not keep the GPUs idle.

*Previous benchmarks:*

When setting the batching parameters, be mindful of the system and gpu memory of the machine you 
are running the script on. Feel free to experiment, but some reasonable `-T` numbers would be 64/128. CPU memory is larger, and increasing `-B` might further improve performance.

Batching Performance Benchmarks:
- Input data: GATK generated WES sample with ~ 90K variants in genome build GRCh37.
- Total predictions made : 174,237
- invitae v2 mainly implements logic to prioritize full batches while predicting 
- settings : 
    - invitae & invitae v2 : B = T = 64
    - invitae v2 optimal : on V100 : B = 4096 ; T = 256 -- on K80/GeForce : B = 4096 ; T = 64

Benchmark results

| Type                                 | Implementation        | Total Time | Speed (predictions / hour) |
|--------------------------------------|-----------------------|------------|----------------------------|
| CPU (intel i5-8365U)<sup>a</sup>     | illumina              | ~100h      | ~1000 pred/h               |
|                                      | invitae               | ~39h       | ~4500 pred/h               |
|                                      | invitae v2            | ~35h       | ~5000 pred/h               |
|                                      | invitae v2 optimal    | ~35h       | ~5000 pred/h               |
| K80 GPU (AWS p2.large)               | illumina</sup>b</sup> | ~25 h      | ~7000 pred/h               |
|                                      | invitae               | 242m       | ~43,000 pred / h           |
|                                      | invitae v2            | 213m       | ~50,000 pred / h           |
|                                      | invitae v2 optimal    | 188 m      | ~56,000 pred / h           |
| GeForce RTX 2070 SUPER GPU (desktop) | illumina</sup>b</sup> | ~10 h      | ~ 17,000 pred/h            |
|                                      | invitae               | 76m        | ~137,000 pred / h          |
|                                      | invitae v2            | 63m        | ~166,000 pred / h          |
|                                      | invitae v2 optimal    | 52m        | ~200,000 pred / h          |
| V100 GPU (AWS p3.xlarge)             | illumina<sup>b</sup>  | ~10h       | ~18,000 pred/h             |
|                                      | invitae               | 78m        | ~135,000 pred / h          |
|                                      | invitae v2            | 54m        | ~190,000 pred / h          |  
|                                      | invitae v2 optimal    | 31 m       | ~335,000 pred / h          |
  

<sup>(a)</sup> : Extrapolated from first 500 variants

<sup>(b)</sup> : Illumina implementation showed a memory leak with the installed versions of tf/keras/.... Values extrapolated from incomplete runs at the point of OOM. 

*Note:* On a p3.8xlarge machine, hosting 4 V100 GPU's, we were able reach 1,379,505 predictions/hour ! This is a nearly linear scale-up.

### Details of SpliceAI INFO field: (TO UPDATE)

| ID     | Description                    |
|--------|--------------------------------|
| ALLELE | Alternate allele               |
| SYMBOL | Gene symbol                    |
| DS_AG  | Delta score (acceptor gain)    |
| DS_AL  | Delta score (acceptor loss)    |
| DS_DG  | Delta score (donor gain)       |
| DS_DL  | Delta score (donor loss)       |
| DP_AG  | Delta position (acceptor gain) |
| DP_AL  | Delta position (acceptor loss) |
| DP_DG  | Delta position (donor gain)    |
| DP_DL  | Delta position (donor loss)    |

Delta score of a variant, defined as the maximum of (DS_AG, DS_AL, DS_DG, DS_DL), ranges from 0 to 1 and can be interpreted as the probability of the variant being splice-altering. In the paper, a detailed characterization is provided for 0.2 (high recall), 0.5 (recommended), and 0.8 (high precision) cutoffs. Delta position conveys information about the location where splicing changes relative to the variant position (positive values are downstream of the variant, negative values are upstream).

### Examples
A sample input file and the corresponding output file can be found at `examples/input.vcf` and `examples/output.vcf` respectively. The output `T|RYR1|0.00|0.00|0.91|0.08|-28|-46|-2|-31` for the variant `19:38958362 C>T` can be interpreted as follows:
* The probability that the position 19:38958360 (=38958362-2) is used as a splice donor increases by 0.91.
* The probability that the position 19:38958331 (=38958362-31) is used as a splice donor decreases by 0.08.

Similarly, the output `CA|TTN|0.07|1.00|0.00|0.00|-7|-1|35|-29` for the variant `2:179415988 C>CA` has the following interpretation:
* The probability that the position 2:179415981 (=179415988-7) is used as a splice acceptor increases by 0.07.
* The probability that the position 2:179415987 (=179415988-1) is used as a splice acceptor decreases by 1.00.

### Frequently asked questions

**1. Why are some variants not scored by SpliceAI?**

SpliceAI only annotates variants within genes defined by the gene annotation file. Additionally, SpliceAI does not annotate variants if they are close to chromosome ends (5kb on either side), deletions of length greater than twice the input parameter ```-D```, or inconsistent with the reference fasta file.

**2. What are the differences between raw (```-M 0```) and masked (```-M 1```) precomputed files?**

The raw files also include splicing changes corresponding to strengthening annotated splice sites and weakening unannotated splice sites, which are typically much less pathogenic than weakening annotated splice sites and strengthening unannotated splice sites. The delta scores of such splicing changes are set to 0 in the masked files. We recommend using raw files for alternative splicing analysis and masked files for variant interpretation.

**3. Can SpliceAI be used to score custom sequences?**

Yes, install SpliceAI and use the following script:  

```python
from keras.models import load_model
from pkg_resources import resource_filename
from spliceai.utils import one_hot_encode
import numpy as np

input_sequence = 'CGATCTGACGTGGGTGTCATCGCATTATCGATATTGCAT'
# Replace this with your custom sequence

context = 10000
paths = ('models/spliceai{}.h5'.format(x) for x in range(1, 6))
models = [load_model(resource_filename('spliceai', x)) for x in paths]
x = one_hot_encode('N'*(context//2) + input_sequence + 'N'*(context//2))[None, :]
y = np.mean([models[m].predict(x) for m in range(5)], axis=0)

acceptor_prob = y[0, :, 1]
donor_prob = y[0, :, 2]
```

### Modifications to Original

**Batching Support** - Invitae (_December 2021_)

* Adds new command line parameters, `--prediction-batch-size` and `--tensorflow-batch-size` to support batching variants to optimize prediction utilization on a GPU 
* Adds a `VCFPredictionBatch` class that manages collection the VCF records, placing them in batches based on the encoded tensor size. Once the batch size is reached, predictions are run in batches, then output is written back in the original order reassembling the annotations for the VCF record. Each VCF record has a lookup key for where each of the ref/alts are within their batches, so it knows where to grab the results during reassembly
* Breaks out code in the existing `get_delta_scores` method into reusable methods used in the batching and the original source code. This way the batching code can utilize the same logic inside that method while still maintaining the original version
* Adds batch utility methods that split up what was all previously done in `get_delta_scores`. `encode_batch_record` handles what was in the first half, taking in the VCF record and generating one-hot encoded matrices for the ref/alts. `extract_delta_scores` handles the second half of the `get_delta_scores` by reassembling the annotations based on the batched predictions
* Adds test cases to run a small file using a generated FASTA reference to test if the results are the same with no batching and with different batching sizes
* Slightly modifies the entrypoint of running the code to allow for easier unit testing. Being able to pass in what would normally come from the argparser

**Multi-GPU support** - Geert Vandeweyer (_November 2022_)

* Offload more code to CPU (eg np to tensor conversion) to *only* perform predictions on the GPU
* Implement queuing system to always have full batches ready for prediction
* Implement new parameter, `--tmpdir` to support a custom tmp folder to store prepped batches
* Implement socket-based client/server approach to scale over multiple GPUs

### Contact
Kishore Jaganathan: kjaganathan@illumina.com

Geert Vandeweyer : geert.vandeweyer@uza.be

Kartik Chundru (This version) : v.chundru@exeter.ac.uk
