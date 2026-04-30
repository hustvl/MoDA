"""
FineWeb dataset (for raw text pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb

This version saves raw text data instead of tokenized samples,
suitable for training with models that handle tokenization during training.

example doc to highlight the structure of the dataset:
{
  "text": "Posted by mattsmith on 20th April 2012\nStraight from...",
  "id": "<urn:uuid:d853d453-196e-4488-a411-efc2b26c40d2>",
  "dump": "CC-MAIN-2013-20",
  "url": "http://nleastchatter.com/philliesphandom/tag/freddy-galvis/",
  "date": "2013-05-18T07:24:47Z",
  "file_path": "s3://commoncrawl/long.../path.../file.gz",
  "language": "en",
  "language_score": 0.9185474514961243,
  "token_count": 594
}
"""
import os
import argparse
import json
from datasets import load_dataset
from tqdm import tqdm

def write_text_shard(filename, documents):
    """
    Saves documents as a JSONL file (one JSON object per line).
    Each line contains the full document with all metadata.
    """
    print(f"writing {len(documents):,} documents to {filename}")
    with open(filename, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc) + "\n")

# ------------------------------------------

parser = argparse.ArgumentParser(description="FineWeb dataset preprocessing (raw text)")
parser.add_argument("-v", "--version", type=str, default="10B", help="Which version of fineweb to use 10B|100B")
parser.add_argument("-s", "--shard_size", type=int, default=10000, help="Number of documents per shard")
parser.add_argument("--min_length", type=int, default=100, help="Minimum text length to include document")
parser.add_argument("--max_length", type=int, default=50000, help="Maximum text length to include document")
args = parser.parse_args()

# FineWeb has a few possible subsamples available
assert args.version in ["10B", "100B"], "version must be one of 10B, 100B"
if args.version == "10B":
    local_dir = "fineweb10B_text"
    remote_name = "sample-10BT"
elif args.version == "100B":
    local_dir = "fineweb100B_text"
    remote_name = "sample-100BT"

# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# download the dataset
print(f"Loading FineWeb dataset: {remote_name}")
fw = load_dataset("HuggingFaceFW/fineweb", name=remote_name, split="train")

def filter_document(doc):
    """Filter documents based on text length and other criteria"""
    text_length = len(doc["text"])
    return args.min_length <= text_length <= args.max_length

def process_document(doc):
    """Process a single document, keeping raw text and metadata"""
    return {
        "text": doc["text"],
        "id": doc.get("id", ""),
        "url": doc.get("url", ""),
        "date": doc.get("date", ""),
        "language": doc.get("language", "en"),
        "language_score": doc.get("language_score", 0.0),
        "token_count": doc.get("token_count", 0),
        "char_count": len(doc["text"])
    }

# Process documents and write output shards
shard_index = 0
current_shard = []
total_docs = 0
filtered_docs = 0

print("Processing documents...")
progress_bar = tqdm(fw, desc="Processing", unit="docs")

for doc in progress_bar:
    total_docs += 1

    # Filter document
    if not filter_document(doc):
        continue

    filtered_docs += 1
    processed_doc = process_document(doc)
    current_shard.append(processed_doc)

    # Check if current shard is full
    if len(current_shard) >= args.shard_size:
        # Determine split (first shard is validation, rest are training)
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"fineweb_{split}_{shard_index:06d}.jsonl")
        write_text_shard(filename, current_shard)
        shard_index += 1
        current_shard = []

    # Update progress bar with filtering info
    progress_bar.set_postfix({
        'filtered': filtered_docs,
        'shards': shard_index,
        'filter_rate': f"{filtered_docs/total_docs:.1%}" if total_docs > 0 else "0%"
    })

# Write any remaining documents as the last shard
if current_shard:
    split = "val" if shard_index == 0 else "train"
    filename = os.path.join(DATA_CACHE_DIR, f"fineweb_{split}_{shard_index:06d}.jsonl")
    write_text_shard(filename, current_shard)
    shard_index += 1

progress_bar.close()

print(f"\nProcessing complete!")
print(f"Total documents processed: {total_docs:,}")
print(f"Documents kept after filtering: {filtered_docs:,}")
print(f"Filter rate: {filtered_docs/total_docs:.1%}")
print(f"Total shards created: {shard_index}")
print(f"Average documents per shard: {filtered_docs/shard_index:.1f}")
print(f"Data saved to: {DATA_CACHE_DIR}")

# Create a metadata file with processing info
metadata = {
    "version": args.version,
    "shard_size": args.shard_size,
    "min_length": args.min_length,
    "max_length": args.max_length,
    "total_documents": total_docs,
    "filtered_documents": filtered_docs,
    "filter_rate": filtered_docs/total_docs,
    "total_shards": shard_index,
    "data_format": "jsonl",
    "description": "FineWeb dataset saved as raw text with metadata"
}

metadata_file = os.path.join(DATA_CACHE_DIR, "metadata.json")
with open(metadata_file, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Metadata saved to: {metadata_file}")
