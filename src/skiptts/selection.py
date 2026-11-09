# Code to transform the large S2ORC dataset into a 2-feature labeled dataset

import gzip
import json
import spacy

nlp = spacy.blank("en")

def process_shard(input_file_path, output_file_path):
    # words, label 1=citation 0=body

    with gzip.open(input_file_path, 'rt', encoding='utf-8') as infile , \
        open(output_file_path, 'a', encoding='utf-8') as outfile:

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
            doc = nlp(body_text)
            tokens = []
            labels = []

            for token in doc:
                # Skip whitespace
                if not token.text.strip():
                    continue

                tokens.append(token.text)
                if is_citation_char[token.idx]:
                    labels.append(0)
                else:
                    labels.append(1)

            datapoint = {
                "tokens": tokens,
                "labels": labels
            }
            outfile.write(json.dumps(datapoint) + "\n")
