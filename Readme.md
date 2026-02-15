<div align="center">

# How often do Answers Change? Estimating Recency Requirements in Question Answering

⏰ **RecencyQA: A benchmark for temporal sensitivity in Question Answering**

<a href="https://github.com/DataScienceUIBK/RecencyQA"><img src="https://img.shields.io/static/v1?label=Dataset&message=GitHub&color=blue&logo=github"></a>
<a href="https://arxiv.org/abs/XXXX.XXXXX"><img src="https://img.shields.io/static/v1?label=Paper&message=ArXiv&color=green&logo=arxiv"></a>
<a href="https://opensource.org/license/apache-2-0"><img src="https://img.shields.io/static/v1?label=License&message=MIT&color=red"></a>

</div>

---


## 📌 Overview

Large Language Models (LLMs) often answer time-sensitive questions using outdated knowledge. However, not all questions require the same level of freshness — some answers change hourly, others yearly, and some never change at all.

**RecencyQA** introduces a principled way to model this temporal behavior through a **Recency–Stationarity Taxonomy** and releases the first dataset that explicitly annotates:

- ✅ Expected answer update frequency (**Recency**)  
- ✅ Context stability of that frequency (**Stationarity**)  
- ✅ Verified temporal contexts inducing different recency interpretations  

This enables fine-grained evaluation of temporal awareness in QA systems beyond binary "fresh vs. outdated" distinctions.

---

## ✨ Highlights

- 🎯 **Novel Taxonomy**: Two-dimensional framework combining **recency** (how often answers change) and **stationarity** (whether this rate is stable)
- 📊 **4,031 Questions**: Open-domain questions with **12 temporal granularity levels** from hourly to permanent
- 🔄 **Context-Aware**: Each question paired with verified temporal contexts enabling **dynamic evaluation**
- 🧪 **Comprehensive Benchmark**: Evaluation across **6 LLMs** with zero-shot, few-shot, and chain-of-thought prompting
- ✅ **Human Validated**: 76% recency accuracy, 78% stationarity accuracy with human annotators

---

## 📁 Dataset

The RecencyQA dataset is available in this repository under `data/`:

📁 **[Dataset/RecencyQA](./Dataset/RecencyQA.json)** → Complete dataset with all annotations

### Dataset Format

Each JSON entry contains:

```json
{
  "q_id": ,
  "question": "",
  "source": "" ,
  "hop_type": "",
  "unique_labels" : [],
  "label_distribution": {},
  "majority_label": "",
  "label_contexts": [{
            "recency_label": "",
            "context": "",
            }
]
  "final_stationary_label":"",
}
```


**Fields:**
- `question`: The question text
- `recency_samples`: 13 independent recency annotations from LLaMA 3.3 (70B)
- `majority_recency`: Primary recency label (majority vote)
- `recency_distribution`: Distribution over all 12 recency classes
- `stationarity`: "stationary" or "non-stationary"
- `temporal_context`: Verified context that grounds the recency interpretation
- `source`: Origin dataset (freshqa/patqa/situatedqa/generated)
- `multi_hop`: Whether the question requires multi-hop reasoning

---


## 🧠 Recency–Stationarity Taxonomy

We characterize temporal sensitivity along **two orthogonal dimensions**:

### 1️⃣ Recency (Expected Time Until Answer Changes)

12 discrete classes ranging from highly volatile to permanent:

| Recency Class | Expected Change Time |
|---------------|---------------------|
| **An-Hour** | Within an hour |
| **A-Few-Hours** | Within a few hours |
| **A-Day** | Within a day |
| **A-Few-Days** | Within a few days |
| **A-Week** | Within a week |
| **A-Few-Weeks** | Within a few weeks |
| **A-Month** | Within a month |
| **A-Few-Months** | Within a few months |
| **A-Year** | Within a year |
| **A-Few-Years** | Within a few years |
| **Many-Years** | After many years |
| **Never** | Not expected to change |

### 2️⃣ Stationarity (Context Dependence)

Whether the recency classification remains stable over time:

- **Stationary**: Recency label is **context-invariant**
  - Example: *"When is the World Technology Summit held?"* → Always annual
  
- **Non-Stationary**: Recency label is **context-dependent**
  - Example: *"Who is leading the Olympic medal table?"*
    - During Olympics: High-frequency (hourly/daily)
    - Between Olympics: Never changes

**Key Examples:**

| Question | Recency | Stationarity | Explanation |
|----------|---------|--------------|-------------|
| *Who is the CEO of Twitter?* | A-Few-Years | Stationary | Low-frequency, stable |
| *What is the inflation rate in the US?* | A-Month | Non-Stationary | May become high-frequency during crises |
| *Who is leading the Olympic medal table?* | Varies | Non-Stationary | High-frequency during Olympics, none between |

---

## 📊 Dataset Statistics

RecencyQA contains **4,031 questions**, each annotated with recency and stationarity labels.

### Dataset Overview

| Metric | Value |
|--------|-------|
| **Total Questions** | 4,031 |
| **Recency Classes** | 12 (from hourly to permanent) |
| **Total Recency Labels** | 5,237 |
| **Stationary Questions** | 2,910 (72.2%) |
| **Non-Stationary Questions** | 1,121 (27.8%) |
| **Single-hop Questions** | 3,161 (78.4%) |
| **Multi-hop Questions** | 870 (21.6%) |
| **Avg. Question Length** | 14.26 words |
| **Avg. Context Length** | 22.22 words |

### Question Sources

| Source | Count |
|--------|------:|
| FreshQA | 453 |
| PATQA | 1,280 |
| SituatedQA | 931 |
| LLM-Generated (Event-based) | 1,367 |
| **Total** | **4,031** |

---

## 🚀 Quick Start

### Clone Repository

```bash
git clone https://github.com/DataScienceUIBK/RecencyQA.git
cd RecencyQA
```

### Load the Dataset

```python
import json

# Load RecencyQA dataset
with open('Dataset/Recencyqa.json', 'r') as f:
    dataset = json.load(f)

# Example: Access first question
question = dataset[0]
print(f"Question: {question['question']}")
print(f"Q_ID: {question['q_id']}")
print(f"Majority Label: {question['majority_label']}")
print(f"Stationarity: {question['final_stationary_label']}")
print(f"Source: {question['source']}")
print(f"Hop Type: {question['hop_type']}")

# Access label contexts
for ctx in question['label_contexts']:
    print(f"Label: {ctx['recency_label']}, Context: {ctx['context']}")
```

### Filter by Properties

```python
# Filter non-stationary questions
non_stationary = [q for q in dataset if q['final_stationary_label'] == 'Non-Stationary']
print(f"Non-stationary questions: {len(non_stationary)}")

# Filter high-frequency recency (hourly/daily updates)
high_frequency = [q for q in dataset 
                  if q['majority_label'] in ['An-Hour', 'A-Few-Hours', 'A-Day']]
print(f"High-frequency questions: {len(high_frequency)}")

# Filter multi-hop questions
multihop = [q for q in dataset if q['hop_type'] == 'Multi-Hop']
print(f"Multi-hop questions: {len(multihop)}")

# Get recency distribution for a question
question = dataset[0]
print(f"Recency distribution: {question['label_distribution']}")
print(f"Unique labels: {question['unique_labels']}")
print(f"Confidence: {question['confidence']}")
print(f"Entropy: {question['entropy']}")

# Filter by source
freshqa_questions = [q for q in dataset if q['source'] == 'freshqa']
print(f"FreshQA questions: {len(freshqa_questions)}")
```
---

## 🔬 What RecencyQA Evaluates

RecencyQA supports three levels of temporal evaluation:

### 1️⃣ Recency Classification
**Can a model infer how often an answer changes from the question alone?**

- Task: Predict one of 12 recency classes without context
- Evaluates: Intrinsic temporal understanding

### 2️⃣ Context Sensitivity
**Does adding temporal context improve performance — especially for non-stationary questions?**

- Task: Compare performance with/without temporal context
- Evaluates: Ability to leverage contextual temporal cues

### 3️⃣ Recency Transition (RL1 → RL2)
**Can a model adapt when the same question requires different recency labels under different contexts?**

- Task: Correctly predict both RL1 (context C1) and RL2 (context C2)
- Evaluates: Dynamic temporal adaptation

---

## 📈 Benchmarking Results (Paper Summary)

We evaluate **6 LLMs** across three prompting strategies:
- **Zero-shot**
- **Few-shot**  
- **Chain-of-Thought (CoT)**

### Models Evaluated

- **Qwen 2.5** (7B, 72B)
- **LLaMA 3** (8B)
- **Mistral** (24B)
- **Gemma 3** (27B)
- **Apertus** (70B)

### Key Findings

✅ **Recency Classification is Challenging**
- Best strict accuracy: **52.05%** (Gemma 3 27B, few-shot)
- Best tolerant accuracy (±1 class): **78.91%**
- Models capture coarse temporal ordering but struggle with fine-grained distinctions

✅ **Context Helps Non-Stationary Questions**
- Non-stationary questions: **up to +46.0% improvement** with context (LLaMA 3.1 8B)
- Stationary questions: Performance often **degrades** or remains unchanged
- Models struggle to determine when context should influence predictions

✅ **Dynamic Adaptation is Difficult**
- Transition accuracy (RL1 → RL2): Only **14.8%** (Gemma 3 27B)
- Per-context accuracy: **45%+**
- Large gap indicates **static temporal reasoning**

### Top Performing Models

| Task | Best Model | Strategy | Accuracy | Tolerant Acc. |
|------|------------|----------|----------|---------------|
| Recency Classification | Gemma 3 27B | Few-shot | 52.05% | 78.91% |
| Context (Non-Stationary) | LLaMA 3.1 8B | Few-shot | 35.1% (+46.0%) | - |
| Recency Transition | Gemma 3 27B | Few-shot | 14.8% | - |

**Conclusion:**  
Current LLMs exhibit largely **static temporal reasoning** and struggle with dynamic recency adaptation.

> Full result tables are available in the paper (Tables 5, 6, 7).

---

## 🏗️ Dataset Construction Pipeline

The dataset construction involves **4 main stages**:

### 1️⃣ Question Sampling
- **Existing datasets**: FreshQA (453), PATQA (1,280), SituatedQA (931)
- **LLM-generated**: Event-based questions using LLaMA 3.3 (70B) (1,367)
- **Total**: 4,031 unique questions after deduplication and filtering

### 2️⃣ Recency Label Generation
- **13 independent annotations** per question using LLaMA 3.3 (70B)
- **Majority voting** determines primary recency label
- **Full distribution** retained for stationarity analysis

### 3️⃣ Stationarity Classification
- **Consensus** from GPT-5.2, Gemini 3 Flash, and Claude Sonnet 4.5
- **Cross-validation** with recency distribution
- Questions with inconsistent labels are **filtered**

### 4️⃣ Temporal Context Generation
- Context generated for each validated recency label
- **Verified** through 13-iteration relabeling (strict majority required)
- **Longest verified context** retained per question

> **Quality Control**: Each step includes rigorous validation to ensure annotation consistency and reliability.

---

## 👥 Human Evaluation Summary

We conducted human evaluation with **6 graduate students** on **240 sampled questions** to validate annotation quality.

### Agreement with Dataset Labels

| Metric | Strict | Tolerant (±1 class) |
|--------|--------|---------------------|
| **Recency Accuracy** | 76% | 81% |
| **Stationarity Accuracy** | 78% | - |

### Quality Ratings (1-5 scale)

| Dimension | Avg. Score | Interpretation |
|-----------|------------|----------------|
| **Question Clarity** | 4.66 | Very clear |
| **Labeling Difficulty** | 2.26 | Easy to moderate |
| **Answering Difficulty** | 2.41 | Easy to moderate |

These results indicate **high annotation quality** and **reliable temporal interpretations**.

---

## 📁 Repository Structure

```
RecencyQA/
├── data/
│   └── recencyqa.json                          # Your dataset
├── prompts/
│   ├── recency_labeling_prompt.txt             # 3 labeling/generation prompts
│   ├── stationarity_labeling_prompt.txt        
│   ├── context_generation_prompt.txt           
│   ├── evaluation_zero_shot_no_context.txt     # 3 evaluation prompts (no context)
│   ├── evaluation_few_shot_no_context.txt      
│   ├── evaluation_cot_no_context.txt           
│   ├── evaluation_zero_shot_with_context.txt   # 3 evaluation prompts (with context)
│   ├── evaluation_few_shot_with_context.txt    
│   └── evaluation_cot_with_context.txt         
├── LICENSE
└── README.md
```

> **Note**: Evaluation scripts and generation code will be released soon.

---

## 🚀 Research Applications

RecencyQA enables research on:

- ✅ **Recency-aware QA systems**
- ✅ **Retrieval-augmented generation (RAG) triggering** — when to retrieve?
- ✅ **Temporal confidence calibration**
- ✅ **Freshness-sensitive ranking**
- ✅ **Temporal drift analysis**
- ✅ **Dynamic multi-hop reasoning**
- ✅ **Recency-based retrieval gating**

---

# 📄 Paper

**Coming Soon**


---

## 📧 Contact

For questions, collaborations, or issues:

- **Bhawna Piryani**: bhawna.piryani@uibk.ac.at

For questions about the dataset or paper, please open an issue on GitHub or contact us directly.

---

**Last Updated:** February 2026
