# from .download_s2orc import download_all_s2orc_shards
# from .pymupdf_poc import parse_and_read
from .selection import *
import glob
from multiprocessing import Pool
import gzip

def main() -> None:
    # download_all_s2orc_shards()

    # parse_and_read("test.pdf")

    # process_shard("s2orc_shards-2026-10-27/s2orc_v2_shard_000.gz", "test.jsonl")

    # paths = [
    #     "s2orc_shards-2026-10-27/s2orc_v2_shard_000.gz",
    #     "s2orc_shards-2026-10-27/s2orc_v2_shard_047.gz",
    #     "s2orc_shards-2026-10-27/s2orc_v2_shard_094.gz",
    #     "s2orc_shards-2026-10-27/s2orc_v2_shard_141.gz",
    #     "s2orc_shards-2026-10-27/s2orc_v2_shard_188.gz",
    #     "s2orc_shards-2026-10-27/s2orc_v2_shard_235.gz",
    #     "s2orc_shards-2026-10-27/s2orc_v2_shard_282.gz",
    #     "s2orc_shards-2026-10-27/s2orc_v2_shard_329.gz",
    # ]
    # process_shards(8, shard_paths=paths)
    # print(count_total_processed_datapoints())

    # split_processed_shards()

    create_subset("dataset/train.jsonl", "dataset/train-100.jsonl", 100)
    create_subset("dataset/val.jsonl", "dataset/val-100.jsonl", 100)