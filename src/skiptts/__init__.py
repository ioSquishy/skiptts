# from .download_s2orc import download_all_s2orc_shards
import os
from .selection import create_dataset
from datasets import DatasetDict

if os.path.isdir("inline-citation-mask") == False:
    create_dataset([
        "s2orc_shards-2026-10-27/s2orc_v2_shard_000.gz",
        "s2orc_shards-2026-10-27/s2orc_v2_shard_047.gz",
        "s2orc_shards-2026-10-27/s2orc_v2_shard_094.gz",
        "s2orc_shards-2026-10-27/s2orc_v2_shard_141.gz",
        "s2orc_shards-2026-10-27/s2orc_v2_shard_188.gz",
        "s2orc_shards-2026-10-27/s2orc_v2_shard_235.gz",
        "s2orc_shards-2026-10-27/s2orc_v2_shard_282.gz",
        "s2orc_shards-2026-10-27/s2orc_v2_shard_329.gz",
    ])

dataset = DatasetDict.load_from_disk("inline-citation-mask")
    
def main() -> None:
    print("Running")
