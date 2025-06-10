
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split


# ================================================
# Training and Evaluation Pipeline for RoBERTa Models (EN & ZH)
# ================================================

# Tokenization preprocessing function
def preprocess_function(examples, tokenizer):
    """
    Tokenizes input text with padding and truncation.

    Parameters:
    - examples (dict): Dictionary with 'text' key.
    - tokenizer: HuggingFace tokenizer.

    Returns:
    - Tokenized batch of inputs.
    """
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

# Evaluation metrics function
def compute_metrics(pred):
    """
    Computes accuracy, precision, recall, and F1 score.

    Parameters:
    - pred: Predictions object from HuggingFace Trainer.

    Returns:
    - Dictionary of metrics.
    """
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, average='weighted'),
        'recall': recall_score(labels, preds, average='weighted'),
        'f1': f1_score(labels, preds, average='weighted')
    }

# Function to train a single model
def train_single_model(train_df, val_df, model_name, model_alias, batch_size=32, epochs=3, fp16=False):
    """
    Trains a RoBERTa-based model on the given dataset.

    Parameters:
    - train_df (pd.DataFrame): Training data with 'text' and 'label'.
    - val_df (pd.DataFrame): Validation data.
    - model_name (str): Pretrained model name or path.
    - model_alias (str): Alias to use for saved model files.
    - batch_size (int): Batch size for training and evaluation.
    - epochs (int): Number of training epochs.
    - fp16 (bool): Whether to use mixed-precision training.

    Returns:
    - val_metrics (dict): Evaluation metrics on validation set.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Convert DataFrames to HuggingFace Datasets
    train_dataset = Dataset.from_pandas(train_df[["text", "label"]].copy())
    val_dataset = Dataset.from_pandas(val_df[["text", "label"]].copy())

    # Tokenize datasets
    train_dataset = train_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)
    val_dataset = val_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)

    # Define output directory for saving the model
    output_dir = f"/content/drive/MyDrive/data/RoBERTa_{model_alias}_model"
    os.makedirs(output_dir, exist_ok=True)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",                  # Evaluate at the end of each epoch
        save_strategy="epoch",                  # Save model at the end of each epoch
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        save_total_limit=1,                     # Keep only the best checkpoint
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir=os.path.join(output_dir, "logs"),
        report_to="none",
        fp16=fp16                               # Enable FP16 training if specified
    )

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    # Start training
    trainer.train()

    # Save final model and tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Evaluate on validation set
    val_metrics = trainer.evaluate()
    print(f"\n📊 Validation Metrics for {model_alias}:")
    for k, v in val_metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    return val_metrics


val_results = []

# Map each language to its corresponding RoBERTa model
model_config = {
    "en": "roberta-base",
    "zh": "hfl/chinese-roberta-wwm-ext"
}

# 🔁 Loop over each dataset (e.g., en_full, zh_mix)
for dataset_key in datasets.keys():
    lang = dataset_key.split("_")[0]  # Extract language code: 'en' or 'zh'
    model_name = model_config[lang]

    print(f"\n🚀 Now training {dataset_key.upper()} using model {model_name}")

    full_df = datasets[dataset_key]
    data_df = full_df[["text", "label"]].copy()

    # ⚡ Step 1: Split into 90% train+val and 10% test
    train_temp_df, test_df = train_test_split(
        data_df,
        test_size=0.1,
        stratify=data_df['label'],
        random_state=42
    )

    # ⚡ Step 2: Further split train_temp into ~80% train and ~10% val
    train_df, val_df = train_test_split(
        train_temp_df,
        test_size=1/9,  # Approx. 11% of total → becomes ~10% of full data
        stratify=train_temp_df['label'],
        random_state=42
    )

    print(f"Data split -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Train and evaluate
    metrics = train_single_model(
        train_df=train_df,
        val_df=val_df,
        model_name=model_name,
        model_alias=dataset_key,
        batch_size=32,
        epochs=3,
        fp16=False
    )

    # Add dataset key to metrics and save
    metrics.update({
        "dataset": dataset_key
    })
    val_results.append(metrics)

    # Save the test split for later evaluation
    save_test_path = f"/content/drive/MyDrive/data/test_sets/RoBERTa_{dataset_key}_test.csv"
    os.makedirs(os.path.dirname(save_test_path), exist_ok=True)
    test_df.to_csv(save_test_path, index=False)
    print(f"✅ Saved test split for {dataset_key} to {save_test_path}")



# save the results
val_results_df = pd.DataFrame(val_results)
val_results_df.to_csv("/content/drive/MyDrive/data/RoBERTa_val_metrics.csv", index=False)
print("\n✅ Saved all validation metrics!")

# ================================================
# Granularity Generalization Test
# ================================================

def preprocess_function(examples, tokenizer):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, average='weighted'),
        'recall': recall_score(labels, preds, average='weighted'),
        'f1': f1_score(labels, preds, average='weighted')
    }

test_dir = "/content/drive/MyDrive/data/test_sets"

test_files = {
    "en_full": os.path.join(test_dir, "RoBERTa_en_full_test.csv"),
    "en_sent": os.path.join(test_dir, "RoBERTa_en_sent_test.csv"),
    "en_mix": os.path.join(test_dir, "RoBERTa_en_mix_test.csv"),
    "zh_full": os.path.join(test_dir, "RoBERTa_zh_full_test.csv"),
    "zh_sent": os.path.join(test_dir, "RoBERTa_zh_sent_test.csv"),
    "zh_mix": os.path.join(test_dir, "RoBERTa_zh_mix_test.csv"),
}

test_datasets = {}

for name, path in test_files.items():
    df = pd.read_csv(path)
    test_datasets[name] = df
    print(f"✅ Loaded {name} - {len(df)} samples")


results = []

# ================================================
# Evaluate RoBERTa on Corresponding Test Sets and Evaluate Granularity Generalization
# ================================================

# 🔁 Loop over all six trained model variants
for model_key in ["en_full", "en_sent", "en_mix", "zh_full", "zh_sent", "zh_mix"]:
    model_path = f"/content/drive/MyDrive/data/RoBERTa_{model_key}_model"

    # Skip evaluation if the model directory doesn't exist
    if not os.path.exists(model_path):
        print(f"[✘] Skipping {model_key} — model not found.")
        continue

    # Load tokenizer and model from the saved directory
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    # Get the language prefix (en or zh) from the model key
    lang_prefix = model_key.split("_")[0]

    # Filter the test sets to only include those that match the model's language
    filtered_test_sets = {k: v for k, v in test_datasets.items() if k.startswith(lang_prefix)}

    # 🔍 Evaluate the current model on all matching test sets
    for test_key, test_df in filtered_test_sets.items():
        print(f"\n🔍 Evaluating {model_key} model on {test_key} test set...")

        # Convert test DataFrame to HuggingFace Dataset format
        eval_dataset = Dataset.from_pandas(test_df[["text", "label"]].copy())

        # Preprocess the text data using the model's tokenizer
        eval_dataset = eval_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)

        # Define evaluation-only training arguments
        eval_args = TrainingArguments(
            output_dir="./eval_tmp",               # Temporary output directory
            per_device_eval_batch_size=32,        # Batch size for evaluation
            report_to="none"                      # Disable logging/reporting
        )

        # Create a Trainer object for evaluation
        trainer = Trainer(
            model=model,
            args=eval_args,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics       # Custom metric function
        )

        # Run prediction/evaluation
        predictions = trainer.predict(eval_dataset)
        metrics = compute_metrics(predictions)

        # Store results in a list of dictionaries
        results.append({
            "Train Set": model_key,
            "Test Set": test_key,
            "Accuracy": round(metrics["accuracy"], 4),
            "Precision": round(metrics["precision"], 4),
            "Recall": round(metrics["recall"], 4),
            "F1": round(metrics["f1"], 4),
        })


# ================================================
# AGT-2025 Generalization Evaluation for Trained Models
# ================================================

import pandas as pd

# --- Load Chinese and English AGT Datasets ---
# Chinese: AI- and human-generated samples
zh_ai_path = "/content/drive/MyDrive/data/AGT2025/AGT2025_fromHC3_chinese_GPT4.5&DSR1.csv"
zh_human_path = "/content/drive/MyDrive/data/AGT2025/hc3_zh_human_sample.csv"

# English: AI- and human-generated samples
en_ai_path = "/content/drive/MyDrive/data/AGT2025/AGT2025_fromHC3_english_GPT4.5&DSR1.xlsx"
en_human_path = "/content/drive/MyDrive/data/AGT2025/hc3_en_human_sample.csv"

# Read files
agt_zh_ai = pd.read_csv(zh_ai_path)
agt_zh_human = pd.read_csv(zh_human_path)
agt_en_ai = pd.read_excel(en_ai_path)
agt_en_human = pd.read_csv(en_human_path)

# Print column names for inspection
print("ZH - AI columns:", agt_zh_ai.columns.tolist())
print("ZH - Human columns:", agt_zh_human.columns.tolist())
print("EN - AI columns:", agt_en_ai.columns.tolist())
print("EN - Human columns:", agt_en_human.columns.tolist())

# --- Validate label values ---
print("ZH - AI label distribution:\n", agt_zh_ai["label"].value_counts())
print("ZH - Human label distribution:\n", agt_zh_human["label"].value_counts())
print("EN - AI label distribution:\n", agt_en_ai["label"].value_counts())
print("EN - Human label distribution:\n", agt_en_human["label"].value_counts())

# Ensure labels are correct: AI=1, Human=0
assert set(agt_zh_ai["label"].unique()) == {1}, "ZH AI label check failed!"
assert set(agt_zh_human["label"].unique()) == {0}, "ZH Human label check failed!"
assert set(agt_en_ai["label"].unique()) == {1}, "EN AI label check failed!"
assert set(agt_en_human["label"].unique()) == {0}, "EN Human label check failed!"

print("✅ Label check passed! All labels are correct.")

# --- Combine into full evaluation sets ---
agt_zh_df = pd.concat([
    agt_zh_ai[["answer", "label"]].rename(columns={"answer": "text"}),
    agt_zh_human[["answer", "label"]].rename(columns={"answer": "text"})
], ignore_index=True)
print(f"✅ Combined Chinese AGT dataset. Samples: {len(agt_zh_df)}")

agt_en_df = pd.concat([
    agt_en_ai[["answer", "label"]].rename(columns={"answer": "text"}),
    agt_en_human[["answer", "label"]].rename(columns={"answer": "text"})
], ignore_index=True)
print(f"✅ Combined English AGT dataset. Samples: {len(agt_en_df)}")

# Ensure all text is string type
agt_en_df['text'] = agt_en_df['text'].astype(str)
agt_zh_df['text'] = agt_zh_df['text'].astype(str)


# ================================================
# Evaluation Setup
# ================================================

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

# --- Preprocessing and metrics ---
def preprocess_function(examples, tokenizer):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, average='weighted'),
        'recall': recall_score(labels, preds, average='weighted'),
        'f1': f1_score(labels, preds, average='weighted')
    }


# ================================================
# Evaluate Each Model on AGT Dataset
# ================================================

model_keys = ["en_full", "en_mix", "en_sent", "zh_full", "zh_mix", "zh_sent"]
results = []

for model_key in model_keys:
    model_path = f"/content/drive/MyDrive/data/{model_key}_model"

    if not os.path.exists(model_path):
        print(f"[✘] Skipping {model_key} — model not found.")
        continue

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    # Select the correct language test set
    test_df = agt_en_df if model_key.startswith("en") else agt_zh_df

    # Tokenize and prepare dataset
    eval_dataset = Dataset.from_pandas(test_df[["text", "label"]].copy())
    eval_dataset = eval_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)

    eval_args = TrainingArguments(
        output_dir="./eval_tmp",  # Temporary directory
        per_device_eval_batch_size=32,
        report_to="none"         # Suppress logging
    )

    trainer = Trainer(
        model=model,
        args=eval_args,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    predictions = trainer.predict(eval_dataset)
    metrics = compute_metrics(predictions)

    results.append({
        "Model": model_key,
        "Accuracy": round(metrics["accuracy"], 4),
        "Precision": round(metrics["precision"], 4),
        "Recall": round(metrics["recall"], 4),
        "F1": round(metrics["f1"], 4),
    })


# ================================================
# Display Results
# ================================================

results_df = pd.DataFrame(results).set_index('Model')

print("\n📊 Evaluation Results on AGT Dataset:")
print(results_df)

# --- Heatmap Visualization ---
plt.figure(figsize=(10, 6))
sns.heatmap(results_df, annot=True, fmt=".4f", cmap="YlGnBu")
plt.title("AGT Dataset Generalization Evaluation")
plt.ylabel("Model")
plt.tight_layout()
plt.show()


# ================================================
# CLONG-2025 (COLING2024 Subset) Generalization Evaluation
# ================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

# --- Step 1: Load JSONL files as DataFrames ---
def read_jsonl(path):
    """
    Reads a .jsonl file and returns a pandas DataFrame.
    """
    with open(path, 'r', encoding='utf-8') as f:
        lines = [json.loads(line.strip()) for line in f]
    return pd.DataFrame(lines)

# Load Chinese test set
zh_coling_path = "/content/drive/MyDrive/data/COLING2024/test_set_multilingual_with_label.jsonl"
zh_coling = read_jsonl(zh_coling_path)
print(f"✅ Loaded {len(zh_coling)} samples from Chinese COLING2024 test set.")

# Load English test set
en_coling_path = "/content/drive/MyDrive/data/COLING2024/test_set_en_with_label.jsonl"
en_coling = read_jsonl(en_coling_path)
print(f"✅ Loaded {len(en_coling)} samples from English COLING2024 test set.")


# ================================================
# Step 2: Clean and Sample Balanced Subsets
# ================================================

# Filter for Chinese only (some multilingual JSONL files may contain others)
zh_coling_filtered = zh_coling[zh_coling["language"] == "Chinese"]

# Sample 2500 AI + 2500 human examples per language (balanced)
zh_sampled = zh_coling_filtered.groupby("label", group_keys=False)\
    .apply(lambda x: x.sample(n=2500, random_state=42)).reset_index(drop=True)

en_sampled = en_coling.groupby("label", group_keys=False)\
    .apply(lambda x: x.sample(n=2500, random_state=42)).reset_index(drop=True)

# Drop missing text and ensure string type
zh_sampled = zh_sampled.dropna(subset=["text"])
en_sampled = en_sampled.dropna(subset=["text"])
zh_sampled["text"] = zh_sampled["text"].astype(str)
en_sampled["text"] = en_sampled["text"].astype(str)


# ================================================
# Step 3: Define Preprocessing and Evaluation Functions
# ================================================

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from transformers.trainer_utils import PredictionOutput
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import os

# Tokenization
def preprocess_function(examples, tokenizer):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

# Metrics computation
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, average='weighted'),
        'recall': recall_score(labels, preds, average='weighted'),
        'f1': f1_score(labels, preds, average='weighted')
    }


# ================================================
# Step 4: Evaluate Each Model on CLONG (COLING2024 Subset)
# ================================================

model_keys = ["en_full", "en_mix", "en_sent", "zh_full", "zh_mix", "zh_sent"]
results = []

for model_key in model_keys:
    model_path = f"/content/drive/MyDrive/data/RoBERTa_{model_key}_model"
    if not os.path.exists(model_path):
        print(f"[✘] Model {model_key} not found. Skipping.")
        continue

    print(f"🔍 Evaluating model: {model_key}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    # Use corresponding test set
    test_df = en_sampled if model_key.startswith("en") else zh_sampled

    # Convert and tokenize test data
    eval_dataset = Dataset.from_pandas(test_df[["text", "label"]].copy())
    eval_dataset = eval_dataset.map(
        lambda batch: preprocess_function(batch, tokenizer),
        batched=True,
        remove_columns=["text"]  # Only remove text, keep label
    )

    eval_args = TrainingArguments(
        output_dir="./eval_tmp",
        per_device_eval_batch_size=32,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=eval_args
    )

    predictions = trainer.predict(eval_dataset)

    # Evaluate with metrics
    if isinstance(predictions, PredictionOutput) and predictions.label_ids is not None:
        metrics = compute_metrics(predictions)
    else:
        metrics = {}

    results.append({
        "Model": model_key,
        "Accuracy": round(metrics.get("accuracy", 0.0), 4),
        "Precision": round(metrics.get("precision", 0.0), 4),
        "Recall": round(metrics.get("recall", 0.0), 4),
        "F1": round(metrics.get("f1", 0.0), 4),
    })


# ================================================
# Step 5: Visualize and Save Results
# ================================================

results_df = pd.DataFrame(results).set_index("Model")

print("\n📊 Generalization Results on CLONG-2025 (COLING Subset):")
print(results_df)

# Save results to CSV
results_df.to_csv("coling_generalization_results.csv")
print("✅ Evaluation results saved to: coling_generalization_results.csv")

# Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(results_df, annot=True, fmt=".4f", cmap="YlGnBu")
plt.title("CLONG-2025 (COLING) Generalization Heatmap")
plt.tight_layout()
plt.show()
