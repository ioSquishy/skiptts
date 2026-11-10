# Code to transform the large S2ORC dataset into a 2-feature labeled dataset

import gzip
import json
import spacy
import glob
import os
from multiprocessing import Pool
import random

nlp = spacy.blank("en")

def process_shard(input_file_path, output_file_path=None):
    if output_file_path is None:
        new_name = os.path.basename(input_file_path).replace("s2orc_v2_", "").replace(".gz", ".jsonl")
        output_file_path = f"processed_shards/processed_{new_name}"

    with gzip.open(input_file_path, 'rt', encoding='utf-8') as infile , \
        open(output_file_path, 'wt', encoding='utf-8') as outfile:

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
            is_citation_char = [False] * len(body_text)

            # Mark citations in mask as True
            for citation in body_citations:
                start_idx = citation.get('start', 0)
                end_idx = citation.get('end', 0)
                is_citation_char[start_idx:end_idx] = [True] * (end_idx-start_idx)

            # Tokenize with blank spacy nlp object to separate by space and punctuation
            # https://spacy.io/usage/processing-pipelines
            doc = nlp(body_text)
            tokens = []
            labels = []

            for token in doc:
                # Skip whitespace
                if not token.text.strip():
                    continue

                tokens.append(token.text)
                if is_citation_char[token.idx]:
                    labels.append(1)
                else:
                    labels.append(0)

            datapoint = {
                "tokens": tokens,
                "labels": labels
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
    pshards = glob.glob("processed_shards/*.jsonl")

    with open("dataset/train.jsonl", mode='wt', encoding='utf-8') as trainfile, \
        open("dataset/val.jsonl", mode='wt', encoding='utf-8') as valfile:

        for shard in pshards:
            with open(shard, mode='rt', encoding='utf-8') as shardfile:
                for datapoint in shardfile:
                    if random.random() < 0.80:
                        trainfile.write(datapoint)
                    else:
                        valfile.write(datapoint)


def create_subset(dataset_path, subset_path, rows):
    with open(dataset_path, mode='r', encoding='utf-8') as superset, \
        open(subset_path, mode='wt', encoding='utf-8') as subset:

        count = 0
        for row in superset:
            if count >= rows:
                return

            subset.write(row)
            count += 1
