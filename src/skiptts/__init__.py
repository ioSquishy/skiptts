# from .download_s2orc import download_all_s2orc_shards
# from .pymupdf_poc import parse_and_read
from .selection import process_shard

def main() -> None:
    # download_all_s2orc_shards()
    # parse_and_read("test.pdf")
    process_shard("s2orc_shards-2026-10-27/s2orc_v2_shard_000.gz", "dataset/full.jsonl")