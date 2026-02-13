## recencyQA  
**How Often Do Answers Change? Estimating Recency Requirements in Question Answering**  
SIGIR 2026

---

## 📌 Overview

Large Language Models (LLMs) often answer time-sensitive questions using outdated knowledge. However, not all questions require the same level of freshness. Some answers change hourly, others yearly, and some never change at all.

**recencyQA** introduces a principled way to model this temporal behavior.

We propose a **Recency–Stationarity Taxonomy** and release **recencyQA**, the first dataset that explicitly annotates:

- ✅ Expected answer update frequency (**Recency**)  
- ✅ Context stability of that frequency (**Stationarity**)  
- ✅ Verified temporal contexts inducing different recency interpretations  

This enables fine-grained evaluation of temporal awareness in QA systems beyond binary "fresh vs outdated" distinctions.

---

## 🧠 Recency–Stationarity Taxonomy

We characterize temporal sensitivity along **two orthogonal dimensions**:

### 1️⃣ Recency (Expected Time Until Answer Changes)

12 discrete classes ranging from highly volatile to permanent:

| Class | Expected Change |
|-------|-----------------|
| An-Hour | Within an hour |
| A-Few-Hours | Within a few hours |
| A-Day | Within a day |
| A-Few-Days | Within a few days |
| A-Week | Within a week |
| A-Few-Weeks | Within a few weeks |
| A-Month | Within a month |
| A-Few-Months | Within a few months |
| A-Year | Within a year |
| A-Few-Years | Within a few years |
| Many-Years | After many years |
| Never | Not expected to change |

---

### 2️⃣ Stationarity

- **Stationary** → Recency class remains stable over time  
- **Non-Stationary** → Recency class depends on context  

Example:

- *Who is the CEO of X?* → Typically stationary  
- *Who is leading the Olympic medal table?* → Non-stationary  

---

## 📊 Dataset Statistics

After verification and filtering:

- **4,031 questions**
- **12 recency classes**
- **2,910 stationary questions**
- **1,121 non-stationary questions**
- Average question length: 14.26 words
- Average context length: 22.22 words

Each question includes:

- 13 recency samples  
- Majority recency label  
- Full recency distribution  
- Stationarity label  
- Verified temporal context  
- Structured JSON format  

---

## 📂 Dataset Access

The dataset is available in the `Dataset` folder:

📁 **[Dataset/RecencyQA](./Dataset/RecencyQA.json)**

The dataset includes:

- Full dataset (.json)
- Train / Validation / Test splits (if applicable)
- Prompt templates used in experiments
- Context generation prompts

---

## 📁 Dataset Format

Each JSON entry follows this structure:



---

## 🔬 What recencyQA Evaluates

recencyQA supports three levels of temporal evaluation:

### 1️⃣ Recency Classification
Can a model infer how often an answer changes from the question alone?

### 2️⃣ Context Sensitivity
Does adding temporal context improve performance — especially for non-stationary questions?

### 3️⃣ Recency Transition (RL1 → RL2)
Can a model adapt when the same question requires different recency labels under different contexts?

---

## 📈 Key Findings

From experiments across multiple LLMs:

- Fine-grained recency classification is difficult (24–52% strict accuracy)
- Tolerant accuracy is significantly higher → models capture coarse temporal ordering
- Context improves performance for non-stationary questions (up to +40%)
- Context can harm performance for stationary questions
- Transition accuracy remains very low → models struggle with dynamic adaptation

**Conclusion:**  
Current LLMs exhibit largely static temporal reasoning.

---

## 🏗 Dataset Construction Pipeline

### 1️⃣ Question Collection
- FreshQA  
- PATQA  
- SituatedQA  
- LLaMA 3.3-generated event-based questions  

### 2️⃣ Recency Labeling
- 13 independent samples per question  
- Majority voting  
- Recency distribution retained  

### 3️⃣ Stationarity Classification
- GPT-5.2  
- Gemini 3 Flash  
- Claude Sonnet 4.5  
- Cross-validation with recency distribution  

### 4️⃣ Context Generation
- Contexts generated to induce specific recency labels  
- Verified via 13 re-classification runs  
- Only strictly consistent contexts retained  

---

## 🚀 Research Applications

recencyQA enables research on:

- Recency-aware QA  
- Retrieval-augmented generation (RAG) triggering  
- Temporal confidence calibration  
- Freshness-sensitive ranking  
- Temporal drift analysis  
- Dynamic multi-hop reasoning  
- Recency-based retrieval gating  

---

## 📜 Citation

If you use recencyQA, please cite:

```Comming Soon
```

---

## 📄 Paper

Paper PDF:

```
Comming Soon
```

---

## 📜 License

Creative Commons Attribution 4.0 International License (CC BY 4.0)

---

## 🤝 Contact

For questions, collaborations, or issues:

- Bhawna Piryani – bhawna.piryani@uibk.ac.at  
- Zehra Mertz – mertz@mef.edu.tr  
- Adam Jatowt – adam.jatowt@uibk.ac.at  

---
