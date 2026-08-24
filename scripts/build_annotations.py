import gzip
import sys
import os

def parse_annotation(in_path, out_path, is_gff3=False):
    print(f"Parsing {in_path} ...")
    transcripts = {}
    
    with gzip.open(in_path, 'rt') as f:
        for line in f:
            if line.startswith('#'): continue
            parts = line.strip('\n').split('\t')
            if len(parts) < 9: continue
            
            chrom = parts[0]
            feature = parts[2]
            start = int(parts[3])
            end = int(parts[4])
            strand = parts[6]
            info = parts[8]
            
            if feature not in ['exon', 'CDS']:
                continue
                
            tx_id = None
            if is_gff3:
                # GFF3 parsing
                for kv in info.split(';'):
                    if '=' in kv:
                        k, v = kv.split('=', 1)
                        if k == 'transcript_id':
                            tx_id = v
                        elif k == 'Parent' and v.startswith('transcript:'):
                            tx_id = v.replace('transcript:', '')
            else:
                # GTF parsing
                for kv in info.split(';'):
                    kv = kv.strip()
                    if kv.startswith('transcript_id '):
                        tx_id = kv.split(' ', 1)[1].strip('"')
                        
            if not tx_id:
                continue
                
            if tx_id not in transcripts:
                transcripts[tx_id] = {
                    'chrom': chrom,
                    'strand': strand,
                    'exons': [],
                    'cds': []
                }
                
            if feature == 'exon':
                transcripts[tx_id]['exons'].append((start, end))
            elif feature == 'CDS':
                transcripts[tx_id]['cds'].append((start, end))
                
    print(f"Writing to {out_path} ...")
    with open(out_path, 'w') as out:
        out.write("#NAME\tCHROM\tSTRAND\tTX_START\tTX_END\tEXON_START\tEXON_END\tCDS_START\tCDS_END\n")
        for tx_id, data in transcripts.items():
            exons = sorted(data['exons'])
            if not exons: continue
            
            # 0-based starts, 1-based ends
            tx_start = exons[0][0] - 1
            tx_end = exons[-1][1]
            
            exon_starts_str = ','.join(str(s - 1) for s, e in exons) + ','
            exon_ends_str = ','.join(str(e) for s, e in exons) + ','
            
            if data['cds']:
                cds_min = min(s for s, e in data['cds'])
                cds_max = max(e for s, e in data['cds'])
                cds_start = cds_min - 1
                cds_end = cds_max
            else:
                cds_start = -1
                cds_end = -1
                
            out.write(f"{tx_id}\t{data['chrom']}\t{data['strand']}\t{tx_start}\t{tx_end}\t{exon_starts_str}\t{exon_ends_str}\t{cds_start}\t{cds_end}\n")
            
    print("Done!")

if __name__ == '__main__':
    gencode_in = "/Users/kartikchundru/resources/gencode.v49.annotation.gff3.gz"
    gencode_out = "spliceai/annotations/gencode.v49.annotation.txt"
    if os.path.exists(gencode_in):
        parse_annotation(gencode_in, gencode_out, is_gff3=True)
    else:
        print(f"File not found: {gencode_in}")
        
    mane_in = "/Users/kartikchundru/resources/MANE.GRCh38.v1.4.ensembl_genomic.gtf.gz"
    mane_out = "spliceai/annotations/MANE.GRCh38.v1.4.ensembl_genomic.txt"
    if os.path.exists(mane_in):
        parse_annotation(mane_in, mane_out, is_gff3=False)
    else:
        print(f"File not found: {mane_in}")
