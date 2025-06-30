# AGT-2025: AI-Generated Text 2025

**AGT-2025** is a custom-constructed test dataset introduced in the thesis *"Detecting Machine-Generated Text Across Granularity, Domains, and LLMs"* (2025). It is designed to evaluate the generalization ability of detection models to outputs from cutting-edge LLMs released in 2025.

---

## 📦 Dataset Structure

AGT-2025 includes **both English and Chinese** samples. Each language subset contains:

- **Human-written texts** sourced from *HC3* and *M4GT*, with no overlap with training data.
- **AI-generated texts** produced by:
  - **GPT-4.5** (via question-based generation)
  - **DeepSeek R1** (via summary-based rewriting)

Each sample includes a binary label:
- `0` = human-written
- `1` = machine-generated

---

## 🛠️ Dataset Construction

AGT-2025 was built using two distinct generation pipelines:

### 1. Question-Based Generation (HC3-style)

- Prompt: Human-written question from HC3  
- Generator: GPT-4.5 or DeepSeek R1  
- Output: QA-style response  

### 2. Summary-Based Rewriting (M4GT-style)

- Input: Long human-written text from M4GT  
- Step 1: Summarized to 10–20 word prompt  
- Step 2: Prompt sent to GPT-4.5 or DeepSeek R1  
- Output: Rewritten version  

To ensure diversity, each prompt was generated in a clean session with no memory or prior history.

---

## 📊 Basic Statistics

| Language | Label  | Count | Mean Words | Std Dev | Max Words |
|----------|--------|-------|------------|---------|------------|
| English  | Human  | 1000  | 343        | 592     | 6606       |
| English  | AI     | 964   | 121        | 85      | 549        |
| Chinese  | Human  | 1000  | 189        | 398     | 6104       |
| Chinese  | AI     | 1080  | 149        | 90      | 998        |

Tokenization:
- English: Whitespace  
- Chinese: Jieba tokenizer

---

## 📁 Example Format (`.jsonl`)

```json
{
  "id": "zh_agt_0748",
  "text": "如何构建大规模语言模型的预训练语料？",
  "label": 1,
  "source": "deepseek-r1",
  "language": "zh"
}
