# from .download_s2orc import download_all_s2orc_shards
import os
from .selection import create_hf_dataset, DATASET_NAME
from .training import train

from datasets import DatasetDict

if os.path.isdir(DATASET_NAME) == False:
    create_hf_dataset([
        "s2orc_shards-2026-10-27/s2orc_v2_shard_000.gz",
        "s2orc_shards-2026-10-27/s2orc_v2_shard_047.gz",
        "s2orc_shards-2026-10-27/s2orc_v2_shard_094.gz",
        "s2orc_shards-2026-10-27/s2orc_v2_shard_141.gz",
        "s2orc_shards-2026-10-27/s2orc_v2_shard_188.gz",
        "s2orc_shards-2026-10-27/s2orc_v2_shard_235.gz",
        "s2orc_shards-2026-10-27/s2orc_v2_shard_282.gz",
        "s2orc_shards-2026-10-27/s2orc_v2_shard_329.gz",
    ])

dataset = DatasetDict.load_from_disk(DATASET_NAME)
    
def main() -> None:
    print("Dataset Loaded")
    # dataset.push_to_hub("ioSquishy/s2orc-inline-citation-mask")
    train(dataset=dataset)