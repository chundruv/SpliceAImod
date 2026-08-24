import os
import sys
import argparse
import logging
import shelve
import tarfile
import shutil
import multiprocessing as mp
import subprocess as sp
from spliceai.batch.data_handlers import VCFWriter
from spliceai.utils import Annotator

# Configure logging
logging.basicConfig(
    format='%(asctime)s %(levelname)s %(name)s: - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def extract_tar(tar_path, extract_path):
    """
    Extracts the tarball using system tar (faster with pigz) 
    or falls back to Python's native tarfile.
    """
    logger.info(f"Extracting {tar_path} to {extract_path}...")
    os.makedirs(extract_path, exist_ok=True)

    # Method 1: Try system tar with pigz for speed
    if shutil.which("tar") and shutil.which("pigz"):
        try:
            cmd = ["tar", "-I", "pigz", "-xf", tar_path, "-C", extract_path]
            sp.run(cmd, check=True)
            logger.info("Extraction complete (via system tar + pigz).")
            return
        except sp.CalledProcessError as e:
            logger.warning(f"System tar failed ({e}), falling back to Python tarfile.")
        except Exception as e:
            logger.warning(f"Unexpected error with system tar ({e}), falling back.")

    # Method 2: Fallback to python tarfile
    try:
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(path=extract_path)
        logger.info("Extraction complete (via Python tarfile).")
    except Exception as e:
        logger.error(f"Failed to extract tar file: {e}")
        sys.exit(1)

def find_shelf_directory(root_path):
    """
    Recursively searches for the directory containing 'shelf_records'.
    Returns (directory, base_path) tuple.
    """
    logger.info(f"Searching for shelf_records in {root_path}...")
    for root, dirs, files in os.walk(root_path):
        for filename in files:
            if "shelf_records" in filename:
                logger.info(f"Found shelf records at: {root}")
                shelf_files = [f for f in files if 'shelf_records' in f]
                logger.info(f"Shelf files: {shelf_files}")
                
                # Determine the correct base path based on file extensions found
                # Priority: .db.dat/.db.dir/.db.bak > .db alone > no extension
                if any('.db.dat' in f or '.db.dir' in f or '.db.bak' in f for f in shelf_files):
                    # Multi-file DBM format: shelf_records.db.{dat,dir,bak}
                    shelf_base = os.path.join(root, 'shelf_records.db')
                    logger.info("Detected multi-file DBM format (.db.dat/.db.dir/.db.bak)")
                elif 'shelf_records.db' in shelf_files and len([f for f in shelf_files if f.startswith('shelf_records')]) == 1:
                    # Single file .db format (likely sqlite or gdbm)
                    shelf_base = os.path.join(root, 'shelf_records.db')
                    logger.info("Detected single-file .db format")
                else:
                    # No extension or other format
                    shelf_base = os.path.join(root, 'shelf_records')
                    logger.info("Using base path without extension")
                
                logger.info(f"Using shelf base path: {shelf_base}")
                return root, shelf_base
    return None, None

def main():
    parser = argparse.ArgumentParser(description='Run VCFWriter from existing tmp files')
    parser.add_argument('-I', '--input_data', required=True, help='Original input VCF file')
    parser.add_argument('-O', '--output_data', required=True, help='Output VCF file path')
    parser.add_argument('-R', '--reference', required=True, help='Reference genome fasta')
    parser.add_argument('-A', '--annotation', default='gencodev49', help='Annotation (gencodev49/MANEv1.4)')
    parser.add_argument('--tar-file', help='Path to .tar.gz file (required if --extracted-dir not provided)')
    parser.add_argument('--extracted-dir', help='Path to already extracted tmp directory (skips extraction)')
    parser.add_argument('-D', '--distance', type=int, default=500, help='Max distance (must match original run)')
    parser.add_argument('--orig-output', action='store_true', help='Use original output format')
    parser.add_argument('--tsv-output', action='store_true', help='Output in TSV format instead of VCF')
    parser.add_argument('--max-memory', type=float, default=None, help='Max memory in GB (overrides auto-detection, useful for SLURM)')
    
    args = parser.parse_args()

    # Validate arguments
    if not args.tar_file and not args.extracted_dir:
        parser.error("Either --tar-file or --extracted-dir must be provided")
    
    if args.tar_file and args.extracted_dir:
        parser.error("Provide only one of --tar-file or --extracted-dir, not both")

    # Determine the root directory to search
    if args.extracted_dir:
        # Use already extracted directory
        logger.info(f"Using pre-extracted directory: {args.extracted_dir}")
        extract_root = os.path.abspath(args.extracted_dir)
        
        if not os.path.exists(extract_root):
            logger.error(f"Extracted directory does not exist: {extract_root}")
            sys.exit(1)
        
        if not os.path.isdir(extract_root):
            logger.error(f"Path is not a directory: {extract_root}")
            sys.exit(1)
    else:
        # Extract from tar file
        extract_root = os.path.join(os.path.dirname(os.path.abspath(args.tar_file)), "extracted_tmp")
        extract_tar(args.tar_file, extract_root)
        
    # 2. Find the correct temporary directory
    tmpdir, shelf_base_path = find_shelf_directory(extract_root)

    if tmpdir is None or shelf_base_path is None:
        logger.error(f"Could not find 'shelf_records' files in {extract_root}.")
        logger.error(f"Directory contents:")
        for root, dirs, files in os.walk(extract_root):
            logger.error(f"  {root}:")
            for f in files[:10]:  # Show first 10 files
                logger.error(f"    - {f}")
        sys.exit(1)

    # Find the actual shelf files (they may have different extensions)
    logger.info("Verifying shelf database files...")
    
    # List all files in tmpdir to debug
    logger.info(f"Files in {tmpdir}:")
    for item in os.listdir(tmpdir):
        if 'shelf' in item.lower():
            full_path = os.path.join(tmpdir, item)
            size = os.path.getsize(full_path) if os.path.isfile(full_path) else 0
            logger.info(f"  - {item} ({size} bytes)")
    
    # Check for various shelf file formats
    possible_extensions = ['', '.db', '.dat', '.dir', '.bak']
    shelf_files_found = []
    for ext in possible_extensions:
        test_path = shelf_base_path + ext
        if os.path.exists(test_path):
            shelf_files_found.append(test_path)
    
    if not shelf_files_found:
        logger.error(f"No shelf files found with base path: {shelf_base_path}")
        logger.error(f"Tried extensions: {possible_extensions}")
        sys.exit(1)
    
    logger.info(f"Found shelf files: {[os.path.basename(f) for f in shelf_files_found]}")

    # Initialize components
    logger.info("Loading Annotator...")
    ann = Annotator(args.reference, args.annotation, cpu=True, load_models=False)
    
    devices = ["cpu"] 
    manager = mp.Manager()
    shared_dict = manager.dict()

    # Keep shelf on disk - don't load into memory to avoid OOM
    logger.info(f"Opening shelf database: {shelf_base_path}")

    # Import dbm to check which backend is being used
    import dbm
    db_type = dbm.whichdb(shelf_base_path)
    logger.info(f"DBM backend detected: {db_type}")

    # Try multiple approaches to open the shelf
    shelf_records = None
    errors = []

    # Approach 1: Direct path as provided
    try:
        logger.info(f"Attempting to open: {shelf_base_path}")
        shelf_records = shelve.open(shelf_base_path, flag='r')
        num_keys = len(shelf_records)
        logger.info(f"✓ Opened successfully with {num_keys} records (keeping on disk)")
        if num_keys == 0:
            shelf_records.close()
            shelf_records = None
            errors.append(f"Shelf opened but contains 0 records")
    except Exception as e:
        errors.append(f"Direct path failed: {e}")

    # Approach 2: Try without .db extension if it exists
    if shelf_records is None and shelf_base_path.endswith('.db'):
        try:
            alt_path = shelf_base_path[:-3]  # Remove .db
            logger.info(f"Attempting without .db extension: {alt_path}")
            shelf_records = shelve.open(alt_path, flag='r')
            num_keys = len(shelf_records)
            logger.info(f"✓ Opened successfully with {num_keys} records (keeping on disk)")
            if num_keys == 0:
                shelf_records.close()
                shelf_records = None
                errors.append(f"Alt path opened but contains 0 records")
        except Exception as e:
            errors.append(f"Without .db extension failed: {e}")

    # Approach 3: Try with .db extension if it doesn't exist
    if shelf_records is None and not shelf_base_path.endswith('.db'):
        try:
            alt_path = shelf_base_path + '.db'
            logger.info(f"Attempting with .db extension: {alt_path}")
            shelf_records = shelve.open(alt_path, flag='r')
            num_keys = len(shelf_records)
            logger.info(f"✓ Opened successfully with {num_keys} records (keeping on disk)")
            if num_keys == 0:
                shelf_records.close()
                shelf_records = None
                errors.append(f"With .db extension opened but contains 0 records")
        except Exception as e:
            errors.append(f"With .db extension failed: {e}")

    if shelf_records is None:
        logger.error("Failed to load shelf records with any approach:")
        for err in errors:
            logger.error(f"  - {err}")
        logger.error(f"\nPlease check that the shelf file is valid:")
        logger.error(f"  File: {shelf_base_path}")
        logger.error(f"  Size: {os.path.getsize(shelf_base_path) if os.path.exists(shelf_base_path) else 'N/A'}")
        sys.exit(1)

    try:    
        logger.info("Initializing VCFWriter...")
        writer = VCFWriter(
            args=args,
            tmpdir=tmpdir,
            devices=devices,
            ann=ann,
            shelf_records=shelf_records,
            shared_dict=shared_dict,
            max_memory_gb=args.max_memory
        )
        
        logger.info(f"Starting {'TSV' if getattr(args, 'tsv_output', False) else 'VCF'} generation...")
        if getattr(args, 'tsv_output', False):
            writer.process_tsv()
        else:
            writer.process()
        
        logger.info("Done! Output written to " + args.output_data)

    except Exception as e:
        logger.error(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Close the shelf database
        if shelf_records is not None and hasattr(shelf_records, 'close'):
            shelf_records.close()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()