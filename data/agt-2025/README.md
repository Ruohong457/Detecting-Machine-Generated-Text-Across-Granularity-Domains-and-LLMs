# AGT-2025: AI-Generated Text 2025

**AGT-2025** is a custom-constructed test dataset introduced in the thesis *"Detecting Machine-Generated Text Across Granularity, Domains, and LLMs"* (2025). It is designed to evaluate the generalization ability of detection models to outputs from cutting-edge LLMs released in 2025.

---

## 📦 Dataset Structure

AGT-2025 includes **both English and Chinese** samples. Each language subset contains:

- **Human-written texts** sourced from *HC3* and *M4GT*, with no overlap with training data.
- **AI-generated texts** produced by both **GPT-4.5** and **DeepSeek R1**, using:
  - **Question-based generation** (from HC3-style prompts)
  - **Summary-based rewriting** (from M4GT-style inputs)


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

To avoid information leakage or prompt contamination, summarization was performed using temporary ChatGPT-4o sessions with no conversation history or memory.

---

## 🧪 Example Cases from AGT-2025

This section shows two real examples of how human-written content was transformed into machine-generated text in AGT-2025, using both prompting strategies.

---

### 🧾 Example 1: Summary-Based Rewriting (English)

**Original Human Text (excerpt from M4GT):**  
Unlike everyone else in this thread I believe you are talking about why the middle seat in the rear seems to have extra cushioning. I sold cars for some time and the reason for that is this: By moving the center person forward, it creates more room for shoulders in the rear...  
*(Full Reddit-style comment continues with clarification, references to Mazda, and illustrations.)*

**Summarized Prompt:**  
Contrary opinion suggests conventional wisdom may be wrong; challenging dominant ideas can lead to new insights or perspectives.

**AI-Generated Response (GPT-4.5 or DeepSeek R1):**  
Contrary opinions are the grit in the oyster of progress—irritating, essential, and often the catalyst for pearls of new understanding. History’s greatest leaps forward began as heresies: Galileo insisting the Earth orbited the sun, Darwin suggesting we share ancestors with apes...  
*(Essay-style reflection continues, with examples from science, ethics, and AI.)*

---

### 🧾 Example 2: Question-Based Generation (English)

**Human Prompt (from HC3):**  
Do blank CDs, VHS Tapes, Mini Discs etc. weigh less than when they are filled with data? Was wondering, could not find a scientific answer. Please explain like I'm five.

**AI-Generated Response (GPT-4.5 or DeepSeek R1):**  
Yes, these types of media generally degrade over time, even if unused. Blank CDs, VHS tapes, and Mini Discs degrade primarily due to factors such as humidity, temperature, exposure to sunlight, and the breakdown of the physical materials they are composed of. CDs and DVDs can degrade because the reflective layer can oxidize, while magnetic media like VHS tapes can lose their magnetization...

