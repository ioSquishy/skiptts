# Code to transform the large S2ORC dataset into a 2-feature labeled dataset

import gzip
import json
import glob
import os
from multiprocessing import Pool
import random

buffer_size = 2 * 1024 * 1024

def process_shard(input_file_path, output_file_path=None):
    if output_file_path is None:
        new_name = os.path.basename(input_file_path).replace("s2orc_v2_", "").replace(".gz", ".jsonl")
        output_file_path = f"processed_shards/processed_{new_name}"

    with gzip.open(input_file_path, 'rt', encoding='utf-8') as infile , \
        open(output_file_path, 'wt', encoding='utf-8', buffering=buffer_size) as outfile:

        for document in infile:
            paper = json.loads(document)

            body = paper.get('body', {})
            body_annotations = body.get('annotations', {})
            body_text = body.get('text', "") # raw body text including headers and annotations
            if not body_text:
                continue
            
            bib_ref_str = body_annotations.get('bib_ref', []) 
            if not bib_ref_str:
                continue
            body_citations = json.loads(bib_ref_str) # contains index ranges of citations in body_text

            clean_body_text = body_text # body text with citations removed
            citation_char_count = 0

            # Boolean mask of document
            is_citation_char = [0] * len(body_text)

            # Mark citations in mask as True
            for citation in body_citations:
                start_idx = citation.get('start', 0)
                end_idx = citation.get('end', 0)
                is_citation_char[start_idx:end_idx] = [1] * (end_idx-start_idx)

            datapoint = {
                "text": body_text,
                "is_citation": is_citation_char
            }
            outfile.write(json.dumps(datapoint) + "\n")

def process_shards(num_processes, shard_paths=[]):
    if len(shard_paths) == 0:
        shard_paths = glob.glob("s2orc_shards-2026-10-27/*.gz")

    with Pool(num_processes) as pool:
        pool.map(process_shard, shard_paths)

def count_datapoints(path):
    count = 0
    with open(path, 'rt', encoding='utf-8') as file:
        for doc in file:
            count += 1
    return count

def count_total_processed_datapoints():
    total_count = 0
    pshards = glob.glob("processed_shards/*.jsonl")
    for shard in pshards:
        total_count += count_datapoints(shard)
    return total_count

def split_processed_shards():
    """80/10/10 train/val/test split"""
    pshards = glob.glob("processed_shards/*.jsonl")

    with open("dataset/train.jsonl", mode='wt', encoding='utf-8', buffering=buffer_size) as trainfile, \
        open("dataset/val.jsonl", mode='wt', encoding='utf-8', buffering=buffer_size) as valfile, \
        open("dataset/test.jsonl", mode='wt', encoding='utf-8', buffering=buffer_size) as testfile:

        for shard in pshards:
            with open(shard, mode='rt', encoding='utf-8', buffering=buffer_size) as shardfile:
                for datapoint in shardfile:
                    if random.random() < 0.80:
                        trainfile.write(datapoint)
                    else:
                        if random.random() < 0.5:
                            valfile.write(datapoint)
                        else:
                            testfile.write(datapoint)

def create_subset(dataset_path, subset_path, rows):
    with open(dataset_path, mode='r', encoding='utf-8', buffering=buffer_size) as superset, \
        open(subset_path, mode='wt', encoding='utf-8', buffering=buffer_size) as subset:

        count = 0
        for row in superset:
            if count >= rows:
                return

            subset.write(row)
            count += 1
