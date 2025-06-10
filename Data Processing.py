
# Step 1: Load and Merge Filtered HC3 Data (English and Chinese)

import pandas as pd
import os

# Load filtered HC3 CSV files and merge train/test

def load_and_merge_hc3(language):
    path_train = f'/content/drive/MyDrive/data/hc3/{language}_train.csv'
    path_test = f'/content/drive/MyDrive/data/hc3/{language}_test.csv'

    # Read train and test files
    train_df = pd.read_csv(path_train)
    test_df = pd.read_csv(path_test)

    # Print basic information
    print(f"{language.upper()} Train Data Loaded: {len(train_df)} records.")
    print(f"{language.upper()} Test Data Loaded: {len(test_df)} records.")

    # Concatenate train and test
    merged_df = pd.concat([train_df, test_df], ignore_index=True)
    print(f"{language.upper()} Merged Data: {len(merged_df)} records.")

    return merged_df

# Load English and Chinese HC3 data
hc3_en_df = load_and_merge_hc3('en')
hc3_zh_df = load_and_merge_hc3('zh')

# Print merged data samples
print("English Merged Data Sample:")
print(hc3_en_df.head())
print("Chinese Merged Data Sample:")
print(hc3_zh_df.head())

# Step 2: Random Sampling and Removal (500 Human Answers from Each Language) for AGT-2025

import random

# Randomly sample 500 human answers from English and Chinese datasets 
def sample_and_remove(data, lang):
    human_data = data[data['label'] == 0]
    sample = human_data.sample(n=500, random_state=42)
    # Save the sampled data to CSV
    sample.to_csv(f'hc3_{lang}_human_sample.csv', index=False, encoding='utf-8-sig')
    print(f"Saved 500 human answers from {lang} data to CSV.")
    # Remove sampled rows from the original data
    remaining_data = data.drop(sample.index)
    return remaining_data

# Perform sampling and removal for both languages
hc3_en = sample_and_remove(hc3_en_df, 'en')
hc3_zh = sample_and_remove(hc3_zh_df, 'zh')

# Print updated dataset sizes
print(f"Updated English Data Size: {len(hc3_en)}")
print(f"Updated Chinese Data Size: {len(hc3_zh)}")

# Step 3: EDA (Exploratory Data Analysis)

import matplotlib.pyplot as plt
import seaborn as sns

# Basic Information Function
def basic_info(data, lang):
    print(f"\nBasic Information for {lang} Data:")
    print(data.info())
    print(data.describe(include='all'))
    print(f"\nLabel Distribution for {lang} Data:")
    print(data['label'].value_counts())

# Text Length Analysis
def text_length_analysis(data, lang):
    data['text_length'] = data['answer'].apply(lambda x: len(str(x)))
    plt.figure(figsize=(8, 4))
    sns.histplot(data['text_length'], bins=30, kde=True)
    plt.title(f'Text Length Distribution for {lang} Data')
    plt.xlabel('Text Length')
    plt.ylabel('Frequency')
    plt.show()

# Random Sample Check
def random_sample_check(data, lang):
    print(f"\nRandom Sample from {lang} Data:")
    print(data.sample(5))

# Data Cleaning
def clean_data(data):
    # Drop duplicates and missing values
    data = data.dropna(subset=['question', 'answer'])
    data = data.drop_duplicates(subset=['question', 'answer'])
    return data

# EDA for English Data
print("\n--- EDA for English Data ---")
hc3_en = clean_data(hc3_en)
basic_info(hc3_en, 'English')
text_length_analysis(hc3_en, 'English')
random_sample_check(hc3_en, 'English')

# EDA for Chinese Data
print("\n--- EDA for Chinese Data ---")
hc3_zh = clean_data(hc3_zh)
basic_info(hc3_zh, 'Chinese')
text_length_analysis(hc3_zh, 'Chinese')
random_sample_check(hc3_zh, 'Chinese')


# Step 4: Load M4GT data (English and Chinese) and inspect structure
import json
# Load M4GT English data
m4gt_en_file_path = '/content/drive/MyDrive/data/m4gt/M4GT_english(sample-removed).json'

try:
    with open(m4gt_en_file_path, 'r', encoding='utf-8') as f:
        m4gt_en_data = json.load(f)
    print("M4GT English data loaded successfully.")
    print(f"Number of records in M4GT English: {len(m4gt_en_data)}")
except Exception as e:
    print(f"Error loading M4GT English data: {e}")

# Inspect the structure of the first few records (English)
print("Sample record from M4GT English:\n", m4gt_en_data[:3])

# Convert to DataFrame and inspect structure
m4gt_en = pd.DataFrame(m4gt_en_data)
print("M4GT English DataFrame info:")
m4gt_en.info()
print("M4GT English DataFrame head:")
print(m4gt_en.head())


# Load M4GT Chinese data
m4gt_zh_file_path = '/content/drive/MyDrive/data/m4gt/M4GT_chinese(sample-removed).json'

try:
    with open(m4gt_zh_file_path, 'r', encoding='utf-8') as f:
        m4gt_zh_data = json.load(f)
    print("M4GT Chinese data loaded successfully.")
    print(f"Number of records in M4GT Chinese: {len(m4gt_zh_data)}")
except Exception as e:
    print(f"Error loading M4GT Chinese data: {e}")

# Inspect the structure of the first few records (Chinese)
print("Sample record from M4GT Chinese:\n", m4gt_zh_data[:3])

# Convert to DataFrame and inspect structure
m4gt_zh = pd.DataFrame(m4gt_zh_data)
print("M4GT Chinese DataFrame info:")
m4gt_zh.info()
print("M4GT Chinese DataFrame head:")
print(m4gt_zh.head())



# Step 5: Combine Dataset
# HC3 English
hc3_en.rename(columns={"answer": "text"}, inplace=True)
hc3_en["dataset"] = "hc3"

# M4GT English
m4gt_en["dataset"] = "m4gt"

# Align columns for merge
combined_en = pd.concat([hc3_en, m4gt_en], ignore_index=True)
combined_en = combined_en.reindex(columns=list(set(hc3_en.columns).union(set(m4gt_en.columns))))
combined_en.fillna("", inplace=True)

# Filter valid labels
combined_en = combined_en[combined_en["label"].isin([0, 1])]


# HC3 Chinese
hc3_zh.rename(columns={"answer": "text"}, inplace=True)
hc3_zh["dataset"] = "hc3"

# M4GT Chinese
m4gt_zh["dataset"] = "m4gt"

# Combine and align
combined_zh = pd.concat([hc3_zh, m4gt_zh], ignore_index=True)
combined_zh = combined_zh.reindex(columns=list(set(hc3_zh.columns).union(set(m4gt_zh.columns))))
combined_zh.fillna("", inplace=True)

# Filter valid labels
combined_zh = combined_zh[combined_zh["label"].isin([0, 1])]


# ================================================
# Split Sentences and Build Full, Sentence-Level, and Mixed Datasets
# ================================================

from collections import Counter
import pandas as pd
import re
import random
from sklearn.utils import shuffle
from nltk.tokenize import sent_tokenize

# Function to split text into sentences
def split_sentences(text, lang="en"):
    """
    Splits input text into sentences.
    
    Parameters:
    - text: str, the input text to split
    - lang: str, language of the text ("en" for English, "zh" for Chinese)

    Returns:
    - A list of sentence strings
    """
    if lang == "zh":
        # Use regular expressions to split Chinese text by sentence-ending punctuation
        sentences = re.split(r"[。！？]", text)
        return [s.strip() for s in sentences if s.strip()]
    else:
        # Use NLTK's sentence tokenizer for English
        return sent_tokenize(text)

# Function to build a dataset with three versions: full, sentence-split, and mixed
def build_balanced_mix_dataset(df, lang="en", seed=11):
    """
    Constructs three versions of a dataset: full text, sentence-level, and a mixed version.

    Parameters:
    - df: pd.DataFrame, the original dataset with at least a 'text' and 'label' column
    - lang: str, the language of the text ("en" or "zh")
    - seed: int, seed for reproducibility

    Returns:
    - full_df: original dataset (shuffled, without 'text_length' column if present)
    - sent_df: dataset where each row is a single sentence
    - mix_df: half original full text + half sentence-level samples, balanced in size
    """
    # Make a copy and shuffle the dataset
    df = df.copy()
    df = shuffle(df, random_state=seed).reset_index(drop=True)

    # Remove the 'text_length' column if it exists
    if "text_length" in df.columns:
        df = df.drop(columns=["text_length"])

    # Full version: retain all rows as-is
    full_df = df.copy()

    # Sentence-level version: split each text into sentences and create a new row for each
    sent_rows = []
    for _, row in df.iterrows():
        for s in split_sentences(row["text"], lang):
            new_row = row.copy()
            new_row["text"] = s  # Replace the full text with the sentence
            sent_rows.append(new_row)
    sent_df = pd.DataFrame(sent_rows)

    # Mixed version: combine half full-text and half sentence-level samples
    half_size = len(df) // 2
    full_half = df.iloc[:half_size].copy()  # First half of full-text samples

    # Create sentence-level rows from the second half
    sent_half_rows = []
    for _, row in df.iloc[half_size:].iterrows():
        for s in split_sentences(row["text"], lang):
            new_row = row.copy()
            new_row["text"] = s
            sent_half_rows.append(new_row)

    # Randomly sample from the sentence-level rows to match the number of full_half rows
    random.seed(seed)
    sent_half_sample = random.sample(sent_half_rows, k=min(len(sent_half_rows), len(full_half)))
    sent_half = pd.DataFrame(sent_half_sample)

    # Combine and shuffle the mixed dataset
    mix_df = pd.concat([full_half, sent_half], ignore_index=True)
    mix_df = shuffle(mix_df, random_state=seed).reset_index(drop=True)

    return full_df, sent_df, mix_df

# Function to print a summary of the dataset
def summarize_dataset(name, df):
    """
    Prints a summary of the dataset including total samples and label distribution.

    Parameters:
    - name: str, name of the dataset (e.g., "full", "sent", or "mix")
    - df: pd.DataFrame, the dataset to summarize
    """
    total = len(df)
    label_counts = Counter(df["label"])
    print(f"=== {name} ===")
    print(f"Total: {total}")
    for label, count in label_counts.items():
        print(f"Label {label}: {count} ({count / total:.2%})")
    print()


import os

# ================================================
# Save Processed Datasets to Google Drive with Optional Sampling
# ================================================

# Create the directory in Google Drive to save datasets
save_dir = "/content/drive/MyDrive/data/"
os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Function to save a DataFrame to a CSV file, with optional downsampling
def save_dataset(df, name):
    """
    Saves a dataset to the specified Google Drive directory.
    If the dataset has more than 50,000 samples, it performs balanced sampling (max 25,000 per class).

    Parameters:
    - df (pd.DataFrame): Dataset to save.
    - name (str): Name to use for the saved file.
    """
    max_samples = 50000  # Maximum number of samples to save
    label_0 = df[df["label"] == 0]
    label_1 = df[df["label"] == 1]

    if len(df) > max_samples:
        print(f"[Sampling] {name}: {len(df)} -> {max_samples} balanced samples")

        # Calculate how many samples to take from each label (maximum half of total)
        half = max_samples // 2
        n0 = min(len(label_0), half)
        n1 = min(len(label_1), half)

        # Adjust if one class has fewer samples than expected
        min_half = min(n0, n1)

        # Randomly sample from each class
        sampled_0 = label_0.sample(n=min_half, random_state=42)
        sampled_1 = label_1.sample(n=min_half, random_state=42)

        # Combine and shuffle the sampled dataset
        df_sampled = pd.concat([sampled_0, sampled_1]).reset_index(drop=True)
        df_sampled = shuffle(df_sampled, random_state=42)
    else:
        print(f"[Full] {name}: {len(df)} samples (no sampling)")
        df_sampled = df

    # Save the dataset as a CSV file
    save_path = os.path.join(save_dir, f"{name.replace(' ', '_').lower()}.csv")
    df_sampled.to_csv(save_path, index=False)
    print(f"✅ Saved to: {save_path}")

# === Save English datasets ===
save_dataset(full_en_df, "EN_Full")  # Full text English dataset
save_dataset(sent_en_df, "EN_Sent")  # Sentence-level English dataset
save_dataset(mix_en_df, "EN_Mix")    # Mixed English dataset

# === Save Chinese datasets ===
save_dataset(full_zh_df, "ZH_Full")  # Full text Chinese dataset
save_dataset(sent_zh_df, "ZH_Sent")  # Sentence-level Chinese dataset
save_dataset(mix_zh_df, "ZH_Mix")    # Mixed Chinese dataset
