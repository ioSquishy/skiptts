import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

from .preprocess import tokenize_and_align_labels, data_collator, tokenizer
from .config import *

id2label = {1: "CITATION", 0: "BODY"}
label2id = {"CITATION": 1, "BODY": 0}

model = AutoModelForTokenClassification.from_pretrained(
    MODEL_CHECKPOINT,
    id2label=id2label,
    label2id=label2id,
)

training_args = TrainingArguments(
    output_dir= f"./checkpoints/{OUTPUT_DIR_PREFIX}-citations",

    # Optimization
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=1000,

    # Batching & GPU performance
    per_device_train_batch_size=TRAINING_BATCH_SIZE,
    per_device_eval_batch_size=64,
    bf16=True,

    # Evaluation & checkpointing
    eval_strategy="steps",
    eval_steps=0.1,
    save_strategy="steps",
    save_steps=0.1,
    num_train_epochs=1.0,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    save_total_limit=2,

    # Logging
    logging_steps=0.02,
    report_to="none",
)

def train(dataset):
    train_subset = dataset["train"].shuffle(seed=42).select(range(TRAIN_ROWS))
    val_subset   = dataset["validation"].shuffle(seed=42).select(range(10_000))

    tokenized_train = train_subset.map(
        tokenize_and_align_labels,
        batched=True,
        num_proc=8,
        load_from_cache_file=False,
        remove_columns=["text", "is_citation"]
    )

    tokenized_val = val_subset.map(
        tokenize_and_align_labels,
        batched=True,
        num_proc=8,
        load_from_cache_file=False,
        remove_columns=["text", "is_citation"]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()
    print(f"Best F1: {trainer.state.best_metric:.4f} | Checkpoint: {trainer.state.best_model_checkpoint}")
    trainer.save_model(f"./models/{OUTPUT_DIR_PREFIX}-citations")


def compute_metrics(eval_predictions):
    logits, labels = eval_predictions
    predictions = np.argmax(logits, axis=-1)

    valid_mask = labels != -100
    true_labels = labels[valid_mask]
    true_predictions = predictions[valid_mask]

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels,
        true_predictions,
        average="binary",
        pos_label=1, # measures precision in predicting citations specifically. \
        # ^ otherwise, model can score highly just guessing non-citation since papers are mostly normal text
        zero_division=0
    )
    accuracy = accuracy_score(true_labels, true_predictions)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy
    }
