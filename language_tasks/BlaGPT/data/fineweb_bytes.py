"""
FineWeb dataset (for byte-level pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb

This version processes the FineWeb dataset for byte-level training,
converting text directly to UTF-8 bytes instead of using BPE tokenization.

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
import multiprocessing as mp
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

def write_datafile(filename, byte_data):
    """
    Saves byte data as a .bin file, for reading in C.
    - First comes a header with 256 int32s
    - The bytes follow, each as a uint8
    """
    assert len(byte_data) < 2**31, "byte count too large" # ~2.1B bytes
    # construct the header
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520 # magic
    header[1] = 1 # version
    header[2] = len(byte_data) # number of bytes after the 256*4 bytes of header (each 1 byte as uint8)
    # construct the bytes numpy array, if not already
    if not isinstance(byte_data, np.ndarray) or not byte_data.dtype == np.uint8:
        bytes_np = np.array(byte_data, dtype=np.uint8)
    else:
        bytes_np = byte_data
    # write to file
    print(f"writing {len(byte_data):,} bytes to {filename}")
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(bytes_np.tobytes())

def tokenize(doc):
    """
    Processes a single document and returns a numpy array of uint8 bytes.
    Adds separator before each document, similar to how eot token is added in original.
    """
    # the special <|endoftext|> separator delimits all documents
    separator = "<|endoftext|>"
    text_with_separator = separator + doc["text"]

    # Convert to UTF-8 bytes
    byte_data = text_with_separator.encode('utf-8')

    # Convert to numpy array of uint8
    bytes_np = np.frombuffer(byte_data, dtype=np.uint8)

    return bytes_np

def main():
    parser = argparse.ArgumentParser(description="FineWeb dataset preprocessing for byte-level training")
    parser.add_argument("-v", "--version", type=str, default="10B", help="Which version of fineweb to use 10B|100B")
    parser.add_argument("-s", "--shard_size", type=int, default=10**8, help="Size of each shard in bytes")
    args = parser.parse_args()

    # FineWeb has a few possible subsamples available
    assert args.version in ["10B", "100B"], "version must be one of 10B, 100B"
    if args.version == "10B":
        local_dir = "fineweb10B_bytes"
        remote_name = "sample-10BT"
    elif args.version == "100B":
        local_dir = "fineweb100B_bytes"
        remote_name = "sample-100BT"

    # create the cache the local directory if it doesn't exist yet
    DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    # download the dataset
    fw = load_dataset("HuggingFaceFW/fineweb", name=remote_name, split="train")

    # process all documents and write output shards, each of shard_size bytes (last shard has remainder)
    nprocs = max(1, os.cpu_count() - 2) # don't hog the entire system
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        # preallocate buffer to hold current shard
        all_bytes_np = np.empty((args.shard_size,), dtype=np.uint8)
        byte_count = 0
        progress_bar = None

        for doc_bytes in pool.imap(tokenize, fw, chunksize=16):
            # is there enough space in the current shard for the new bytes?
            if byte_count + len(doc_bytes) < args.shard_size:
                # simply append bytes to current shard
                all_bytes_np[byte_count:byte_count+len(doc_bytes)] = doc_bytes
                byte_count += len(doc_bytes)
                # update progress bar
                if progress_bar is None:
                    progress_bar = tqdm(total=args.shard_size, unit="bytes", desc=f"Shard {shard_index}")
                progress_bar.update(len(doc_bytes))
            else:
                # write the current shard and start a new one
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(DATA_CACHE_DIR, f"fineweb_{split}_{shard_index:06d}.bin")
                # split the document into whatever fits in this shard; the remainder goes to next one
                remainder = args.shard_size - byte_count
                progress_bar.update(remainder)
                all_bytes_np[byte_count:byte_count+remainder] = doc_bytes[:remainder]
                write_datafile(filename, all_bytes_np)
                shard_index += 1
                progress_bar = None
                # populate the next shard with the leftovers of the current doc
                all_bytes_np[0:len(doc_bytes)-remainder] = doc_bytes[remainder:]
                byte_count = len(doc_bytes)-remainder

        # write any remaining bytes as the last shard
        if byte_count != 0:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"fineweb_{split}_{shard_index:06d}.bin")
            write_datafile(filename, all_bytes_np[:byte_count])

    print(f"Dataset processing complete. {shard_index + 1} shards created in {DATA_CACHE_DIR}")

if __name__ == "__main__":
    main()
