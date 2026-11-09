# from .download_s2orc import download_all_s2orc_shards
from .pymupdf_poc import parse_and_read

def main() -> None:
    # download_all_s2orc_shards()
    parse_and_read("test.pdf")