from transformers import AutoTokenizer, DataCollatorForTokenClassification

MODEL_CHECKPOINT = "jhu-clsp/ettin-encoder-32m"
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["text"],
        truncation=True,
        max_length=512, # truncate examples over 512 tokens (rare with paragraph splitting)
        return_offsets_mapping=True
    )

    labels = []

    for i, offset_mapping in enumerate(tokenized_inputs["offset_mapping"]):
        char_mask = examples["is_citation"][i]
        token_labels = []

        for start_char, end_char in offset_mapping:
            # special tokens
            if start_char == end_char == 0:
                token_labels.append(-100)
            else: 
                token_labels.append(int(1 in char_mask[start_char:end_char]))

        labels.append(token_labels)

    tokenized_inputs["labels"] = labels
    tokenized_inputs.pop("offset_mapping")
    return tokenized_inputs