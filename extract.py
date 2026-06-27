import json
import os
from Bio import Entrez, SeqIO

# ─────────────────────────────────────────────────────────────────────────
# CONFIGURATION & INITIALIZATION
# ─────────────────────────────────────────────────────────────────────────
# Crucial: NCBI requires an email address to trace API usage.
Entrez.email = "your.email@example.com"  

# Define your input file and the target output directory
JSON_INPUT_FILE = "testing_sources.json"
OUTPUT_DIR = "ncbi_extracted_sequences"

def setup_environment(directory_name):
    """Creates the target directory if it doesn't exist."""
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
        print(f"[✔] Created output directory: '{directory_name}'")
    else:
        print(f"[*] Output directory '{directory_name}' already exists.")

def parse_sources(filepath):
    """Loads and returns sequence records from the provided JSON file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing required metadata file: {filepath}")
        
    with open(filepath, 'r') as file:
        try:
            data = json.load(file)
            print(f"[✔] Successfully parsed {len(data)} metadata targets from {filepath}")
            return data
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON file {filepath}: {e}")

def fetch_and_export_records(sources, target_directory):
    """
    Connects to NCBI Entrez, pulls down genomic blocks with snapped codon-level 
    coordinates, and saves them to local disk as .fasta and .txt.
    """
    print("\nStarting sequence extraction loop from NCBI Nucleotide database...")
    
    for idx, source in enumerate(sources, 1):
        name = source.get("name", f"Unknown_Species_{idx}").replace(" ", "_").replace("(", "").replace(")", "")
        ncbi_id = source.get("id")
        raw_start = source.get("start")
        raw_stop = source.get("stop")
        clade = source.get("clade", "unclassified")
        
        if not ncbi_id:
            print(f"[!] Skipping record index {idx}: No valid NCBI RefSeq ID provided.")
            continue
            
        print(f"\n[{idx}/{len(sources)}] Processing: {name} ({ncbi_id})")
        
        # Build standard API arguments
        fetch_args = {
            "db": "nucleotide",
            "id": ncbi_id,
            "rettype": "gbwithparts",
            "retmode": "text"
        }
        
        # Mirror alignment logic to preserve codon boundaries
        if raw_start is not None and raw_stop is not None:
            raw_start = int(raw_start)
            raw_stop = int(raw_stop)
            
            # Snap start coordinate down to the nearest multiple of 3
            aligned_start = raw_start - (raw_start % 3)
            # Snap stop coordinate up to guarantee length remains a multiple of 3
            aligned_stop = raw_stop + ((3 - (raw_stop % 3)) % 3)
            
            fetch_args["seq_start"] = aligned_start
            fetch_args["seq_stop"] = aligned_stop
            print(f"    -> Aligning coordinates: {raw_start}-{raw_stop} snapped to {aligned_start}-{aligned_stop}")
        else:
            aligned_start, aligned_stop = "Full_Sequence", "Full_Sequence"

        # Construct safe system filepaths
        fasta_path = os.path.join(target_directory, f"{name}.fasta")
        text_path = os.path.join(target_directory, f"{name}.txt")

        # Execute remote Entrez API fetch
        try:
            print(f"    -> Querying Entrez efetch for ID {ncbi_id}...")
            with Entrez.efetch(**fetch_args) as handle:
                # Read the response directly as a BioPython SeqRecord object
                record = SeqIO.read(handle, "genbank")
            
            # 1. Export as FASTA
            with open(fasta_path, "w") as fasta_file:
                SeqIO.write(record, fasta_file, "fasta")
            print(f"    [✔] Saved raw sequence FASTA to: {fasta_path}")
            
            # 2. Export structural features, metadata, and description details to a TXT file
            with open(text_path, "w") as txt_file:
                txt_file.write("=" * 80 + "\n")
                txt_file.write(f"NCBI EXTRACTION LOG & METADATA PROFILE: {record.name}\n")
                txt_file.write("=" * 80 + "\n\n")
                txt_file.write(f"Target Given Name   : {source.get('name')}\n")
                txt_file.write(f"Assigned Clade      : {clade}\n")
                txt_file.write(f"NCBI Accession ID   : {record.id}\n")
                txt_file.write(f"Description line    : {record.description}\n")
                txt_file.write(f"Extracted Length    : {len(record.seq)} base pairs\n")
                txt_file.write(f"Requested Start-Stop: {raw_start} to {raw_stop}\n")
                txt_file.write(f"Aligned Start-Stop  : {aligned_start} to {aligned_stop}\n\n")
                
                txt_file.write("-" * 80 + "\n")
                txt_file.write("GENE FEATURE LOCATIONS MAPPED WITHIN THIS SLICE\n")
                txt_file.write("-" * 80 + "\n")
                
                # Enumerate sub-features contained inside the region (e.g., CDS, exons)
                cds_count = 0
                for feature in record.features:
                    if feature.type == "CDS":
                        cds_count += 1
                        loc = feature.location
                        strand = "+" if loc.strand == 1 else "-"
                        txt_file.write(
                            f"Feature #{cds_count} [{feature.type}]: "
                            f"Slice Relative Coordinates: {int(loc.start)} -> {int(loc.end)} | "
                            f"Strand Direction: [{strand}]\n"
                        )
                        # Extract product or gene information tags if visible
                        if "gene" in feature.qualifiers:
                            txt_file.write(f"  └── Gene Symbol: {feature.qualifiers['gene'][0]}\n")
                        if "product" in feature.qualifiers:
                            txt_file.write(f"  └── Product    : {feature.qualifiers['product'][0]}\n")
                            
                if cds_count == 0:
                    txt_file.write("No distinct CDS (Coding Sequences) annotated inside this sliced window.\n")
                    
            print(f"    [✔] Saved structured profile text metadata to: {text_path}")

        except Exception as err:
            print(f"    [!] Critical failure pulling or digesting entry {ncbi_id}: {err}")

# ─────────────────────────────────────────────────────────────────────────
# MAIN METHOD EXECUTION ROUTINE
# ─────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("NCBI Automated Bulk Sequence Fetcher and Exporter")
    print("=" * 70)
    
    # Run structural directory and file sanity verification
    setup_environment(OUTPUT_DIR)
    
    try:
        source_targets = parse_sources(JSON_INPUT_FILE)
        fetch_and_export_records(source_targets, OUTPUT_DIR)
        print("\n\n[✔] Process complete. All valid outputs stored in folder:", OUTPUT_DIR)
    except Exception as general_error:
        print(f"\n[!] Pipeline halted due to unexpected error: {general_error}")

if __name__ == '__main__':
    main()