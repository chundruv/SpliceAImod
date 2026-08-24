# Original source code modified to add prediction batching support by Invitae in 2021.
# Modifications copyright (c) 2021 Invitae Corporation.
# Converted to PyTorch with GPU optimizations

from setuptools import setup
import io


setup(name='spliceai',
      description='SpliceAI: A deep learning-based tool to identify splice variants (PyTorch GPU-Optimized)',
      long_description=io.open('README.md', encoding='utf-8').read(),
      long_description_content_type='text/markdown',
      version='1.3.2',
      author='Kishore Jaganathan',
      author_email='kishorejaganathan@gmail.com',
      license='GPLv3',
      url='https://github.com/illumina/SpliceAI',
      # spliceai.models holds the single definition of the model architecture
      # (spliceai.utils imports SpliceAI/create_spliceai_model from it) plus
      # the CPU graph-rewrite pass, so it must be installed, not just shipped
      # as package_data.
      packages=['spliceai', 'spliceai.batch', 'spliceai.models'],
      install_requires=[
          'torch>=2.0.0',          # PyTorch 2.0+ for torch.compile support
          'pyfaidx>=0.5.0',
          'pysam>=0.10.0',
          'numpy>=1.14.0',
          'pandas>=0.24.2',
          'psutil>=5.8.0',         # For memory/CPU monitoring
      ],
      extras_require={
          'cpu': [],  # PyTorch CPU is in base requirements
          'gpu': [
              'nvidia-ml-py>=11.0.0',  # For GPU monitoring
          ],
      },
      package_data={'spliceai': ['annotations/grch37.txt',
                                 'annotations/grch38.txt',
                                 'annotations/gencode.v49.annotation.txt',
                                 'annotations/MANE.GRCh38.v1.4.ensembl_genomic.txt',
                                 'models/spliceai1.pt',
                                 'models/spliceai2.pt',
                                 'models/spliceai3.pt',
                                 'models/spliceai4.pt',
                                 'models/spliceai5.pt']},
      entry_points={'console_scripts': ['spliceai=spliceai.__main__:main']},
      test_suite='tests',
      python_requires='>=3.8',
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          'Programming Language :: Python :: 3.10',
          'Programming Language :: Python :: 3.11',
          'Programming Language :: Python :: 3.12',
          'Topic :: Scientific/Engineering :: Bio-Informatics',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
      ],
)