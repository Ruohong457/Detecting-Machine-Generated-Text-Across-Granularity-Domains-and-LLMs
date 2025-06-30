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
Unlike everyone else in this thread I believe you are talking about why the middle seat in the rear seems to have extra cushioning.  I sold cars for some time and the reason for that is this:  By moving the center person forward, it creates more room for shoulders in the rear.  people are wider in the shoulders than they are in the hips, so this makes it easier and more comfortable to seat three people side by side than if they were all shoulder to shoulder.

Obligatory thanks for the gold edit!  Apparently I was the first person who understood what the OP was asking.

Edit for clarification because this seems so controversial.  What OP is talking about is the additional cushioning in the middle seat like found [here](_URL_0_) which he confirms in the reply to this post.  At no point in the post does the OP mention anything about the floor.  yes there is a bump there but that's there for obvious reasons of clearance for undercarriage materials such as strengthening members, exhaust, and driveshaft/transmission.  I worked for Mazda for almost 4 years and met with Jim Sullivan who is the head of Mazda North America and this is his explanation of the seat's shape.

Hope this clears things up!

FURTHER EDIT:

Here's an illustration of the rear seat from above.  Ovals are humans

_URL_1_

**Summarized Prompt:**  
Contrary opinion suggests conventional wisdom may be wrong; challenging dominant ideas can lead to new insights or perspectives.

**AI-Generated Response (GPT-4.5 or DeepSeek R1):**  
Contrary opinions are the grit in the oyster of progress—irritating, essential, and often the catalyst for pearls of new understanding. History’s greatest leaps forward began as heresies: Galileo insisting the Earth orbited the sun, Darwin suggesting we share ancestors with apes, or Einstein unraveling Newton’s clockwork universe. These weren’t just “alternative facts” but seismic shifts in perspective that cracked open rigid paradigms.

Conventional wisdom calcifies not because it’s inherently wrong, but because comfort breeds complacency. Take medicine: for decades, ulcers were blamed on stress, until two maverick researchers—Barry Marshall and Robin Warren—gulped bacteria-laden broth to prove H. pylori was the true culprit. Their Nobel Prize-winning work upended textbooks and saved millions, but only after years of ridicule.

Challenging dominant ideas acts as intellectual antivirus. When Rachel Carson questioned DDT’s safety in Silent Spring, she didn’t just critique pesticides—she exposed humanity’s arrogance in dominating nature. Similarly, modern critiques of AI ethics—like Timnit Gebru’s warnings about biased algorithms—force tech giants to confront the hidden costs of their “progress.”

Yet contrarianism isn’t rebellion for its own sake. The key lies in marrying skepticism with rigor. Consider the flat-Earth theory: while it challenges mainstream cosmology, it collapses under basic physics. True insight emerges when dissent is disciplined—like the physicists who poke at dark matter’s mysteries, not to dismiss it, but to refine or replace it with something better.

This friction between tradition and innovation isn’t a bug of human thought—it’s the feature. Every paradigm shift starts as a whisper in the margins: the economist questioning infinite growth on a finite planet, the artist redefining beauty outside gallery walls, the teenager coding a blockchain alternative to legacy banks.

So next time someone dismisses your “crazy” idea, remember: today’s heresy is often tomorrow’s orthodoxy. The trick is to argue not just with passion, but with proof—and to listen, even when the crowd roars. After all, the Earth feels flat, the sun seems to rise, and sometimes, the crazy idea? It’s just the truth nobody’s ready to hear yet.

---

### 🧾 Example 2: Question-Based Generation (English)

**Human Prompt (from HC3):**  
Do blank CDs, VHS Tapes, Mini Discs etc. weigh less than when they are filled with data? Was wondering, could not find a scientific answer. Please explain like I'm five.

**AI-Generated Response (GPT-4.5 or DeepSeek R1):**  
Yes, these types of media generally degrade over time, even if unused. Blank CDs, VHS tapes, and Mini Discs degrade primarily due to factors such as humidity, temperature, exposure to sunlight, and the breakdown of the physical materials they are composed of. CDs and DVDs can degrade because the reflective layer can oxidize, while magnetic media like VHS tapes can lose their magnetization. Proper storage can significantly slow down the process, but eventual degradation over decades is typical.


