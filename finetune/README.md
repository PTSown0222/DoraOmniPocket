# 🚀 State-of-the-Art LLM Fine-Tuning Framework & Benchmarks

A comprehensive guide aggregating top foundation models (Backbones), real-world tasks, and standard benchmarks for training and fine-tuning Large Language Models (LLMs).

---

## 🛠️ 1. Backbone Models for Fine-Tuning (SOTA Models)

Below is a curated list of open-weights and open-source models widely favored by the community for fine-tuning techniques (LoRA, QLoRA, Full Fine-tuning):

| Backbone Model | Organization / Author | Architecture / Parameters | License Type | Key Strengths | Optimal Use Cases |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **LLaMA (LLaMA 3 / 3.1 / 3.3)** | Meta AI | Dense (8B, 70B, 405B) | Llama Community License | Vast ecosystem support, strong reasoning, massive context length (128K+) | General tasks, Domain Adaptation, general-purpose instruction tuning |
| **Qwen (Qwen 2.5 / Qwen 3 series)** | Alibaba Cloud | Dense & MoE (0.5B - 235B) | Apache 2.0 | Excellent multilingual capabilities, math, coding, and structured JSON output | Commercial applications, multilingual systems, On-device / Edge AI |
| **DeepSeek (V3 / R1 / V4 series)** | DeepSeek AI | MoE / Hybrid Reasoning | MIT License | Outstanding step-by-step reasoning (Chain-of-Thought) and code generation at low cost | Coding automation, complex mathematics, autonomous reasoning agents |
| **Mistral / Mixtral** | Mistral AI | Dense & MoE (7B, 8x7B, 8x22B) | Apache 2.0 | Fast inference, hardware efficiency, strong support for European languages | On-Premise deployment, enterprise solutions with high privacy requirements |
| **Gemma (Gemma 2 / 4)** | Google | Dense (2B, 9B, 27B, 31B) | Gemma Terms of Use | Compact size, punches above its weight class, built-in safety features | On-device deployment (Laptops, Mobile, Edge Devices) |
| **Phi (Phi-3.5 / Phi-4)** | Microsoft | Dense Small Models (3.8B - 14B) | MIT License | Trained on high-quality synthetic "textbook" data, fast response latency | Narrow tasks, Edge Computing, latency-critical applications |
| **GLM (GLM-4 / GLM-5 series)** | Zhipu AI / Z.ai | Dense & MoE | MIT License | Ultra-long context windows (Up to 1M tokens), long-horizon agentic workflows | Ultra-long document analysis, complex task automation |

> **💡 Recommended Fine-Tuning Frameworks:** `Unsloth` (speed & VRAM optimization), `Axolotl`, `Llama-Factory`, `TRL (HuggingFace)`, `DeepSpeed`.

---

## 🎯 2. Tasks & Real-World Applications

A breakdown of fundamental to advanced tasks commonly tackled across industry and research domains:

| Task Category | Detailed Task Name | Input / Output Format | Description & Real-World Applications |
| :--- | :--- | :--- | :--- |
| **Core NLP** | Text Classification & Sentiment Analysis | Text -> Label / Score | News categorization, customer sentiment analysis (CRM, E-commerce reviews). |
| | Named Entity Recognition (NER) | Text -> Entities | Automated information extraction (Names, Emails, Phone Numbers, Dates, Locations). |
| | Text Summarization | Long Text -> Short Summary | Summarizing long reports, meeting minutes, news digests, and research papers. |
| | Machine Translation | Text (Lang A) -> Text (Lang B) | Multilingual translation, domain-specific translation (e.g., medical/legal). |
| **QA & Chat** | Single-turn / Multi-turn QA | Text / Query -> Text Answer | Customer service chatbots, enterprise internal knowledge bots. |
| | Retrieval-Augmented Generation (RAG) | Query + Context -> Grounded Answer | Searching and answering questions grounded on private corporate knowledge bases. |
| **Recommendation & Safety** | Recommendation System | User Profile / Query -> Items List | Recommending products, movies, or articles based on user preference and context. |
| | Content Moderation | Text / Image -> Safe / Unsafe | Detecting harmful content, spam, hate speech, and policy violations on social platforms. |
| **Coding & Engineering** | Code Generation & Completion | Prompt / Docstring -> Code | Writing functions, autocompleting code blocks, acting as a developer copilot. |
| | Code Refactoring & Bug Fixing | Buggy Code -> Clean / Fixed Code | Identifying bugs, optimizing program performance, and improving code structure. |
| | Natural Language to SQL (NL2SQL) | Natural Language -> SQL Query | Translating natural language questions into database queries. |
| **Agents & Automation** | Tool / Function Calling | Prompt -> API Call Payload | Executing external actions via APIs (Sending emails, scheduling, weather lookup, calculations). |
| | Agentic Task Planning | Goal -> Multi-step Action Execution | Breaking down complex goals into sub-tasks and executing them autonomously. |
| **Multimodal** | Image Captioning & Visual QA (VQA) | Image + Text -> Text | Describing images, interpreting charts/diagrams, document OCR with LLMs. |
| **Domain-Specific** | Medical & Clinical QA | Medical Query -> Medical Advice | Assisting in clinical decision support and summarizing medical charts (requires strict validation). |
| | Financial Analysis & IE | Financial Report -> Structured Table | Extracting financial ratios, assessing investment risks, analyzing earnings reports. |
| | Legal Document Analysis | Contract Text -> Risk / Clause Extraction | Contract auditing, clause extraction, legal precedent lookup. |

---

## 📊 3. Evaluation Benchmarks

Standard benchmarks used to measure LLM performance before and after fine-tuning:

| Benchmark Name | Evaluation Area | Test Format | Benchmark Objective | Current Status |
| :--- | :--- | :--- | :--- | :--- |
| **MMLU / MMLU-Pro** | General Knowledge (57+ subjects) | Multiple Choice (4 or 10 options) | Evaluates breadth of knowledge from elementary to professional levels. | MMLU is largely saturated; MMLU-Pro is now the preferred harder standard. |
| **GPQA (GPQA Diamond)** | Domain Expert Science | PhD-level Multiple Choice | Tests deep reasoning in Biology, Chemistry, and Physics. | Considered the gold standard for expert-level scientific reasoning. |
| **SQuAD / SQuAD v2** | Contextual Question Answering | Extractive QA / Unanswerable detection | Tests reading comprehension and ability to recognize unanswerable questions (v2). | Fundamental baseline for basic RAG and QA models. |
| **GSM8K** | Elementary Mathematics | Math word problems with CoT | Evaluates basic multi-step mathematical reasoning logic. | SOTA models achieve >95%; nearing saturation. |
| **MATH / AIME** | Advanced Competition Mathematics | Free-response / Proof problems | Assesses complex mathematical problem-solving and deep logical reasoning. | Primary benchmark for math-specialized and reasoning-focused models. |
| **HumanEval / HumanEval+** | Python Code Generation | Generating Python functions from docstrings | Measures Pass@1 accuracy for code correctness. | HumanEval+ adds rigorous unit tests to prevent memorization/cheating. |
| **SWE-bench / SWE-bench Verified** | Real-world Software Engineering | Resolving GitHub issues directly on repos | Evaluates ability to read large codebases, modify files, and fix real bugs. | The highest standard for evaluating Autonomous Coding Agents. |
| **LiveCodeBench** | Competitive Programming | Continuously updated coding problems | Mitigates data contamination by collecting fresh problems from LeetCode/Codeforces. | Highly regarded for contamination-free coding evaluation. |
| **BFCL (Berkeley Function Calling)** | Function Calling & Tool Use | JSON Output / Execution Trace | Evaluates API call generation accuracy, parallel calling, and multi-turn tool use. | Standard benchmark for evaluating Tool-using AI Agents. |
| **MedQA / BioASQ** | Healthcare & Biomedical Science | Medical Board Exam questions / Medical QA | Measures accuracy in medical knowledge and safe diagnostic suggestions. | Mandatory benchmark for fine-tuning medical domain models. |
| **LegalBench / FinQA** | Law & Financial NLP | Multiple Choice & Table Extraction | Tests contract comprehension, risk detection, and numerical financial reasoning. | Specialized benchmarks for enterprise domain adaptation. |
| **LMSYS Chatbot Arena (Arena Elo)** | Real-world Human Preference | Blind A/B Crowdsourced Testing | Measures human preference win-rates across diverse real-user prompts. | Widely considered the most ecologically valid real-world LLM leaderboard. |

---
## Vietnamese-specific benchmarks for LLM fine-tuning and evaluation:

| Benchmark Name | Evaluation Area | Test Format | Benchmark Objective | Current Status |
| :--- | :--- | :--- | :--- | :--- |
| **VMLU (Vietnamese MMLU)** | Vietnamese Multitask Understanding | Multiple Choice across 58 subjects | Evaluates general knowledge and reasoning across STEM, Humanities, and Social Sciences in Vietnamese. | Primary gold-standard multitask benchmark for Vietnamese foundation models. |
| **ViQuAD / ViMM** | Vietnamese QA & Multimodal | Multiple Choice & Image + Text | Evaluates Vietnamese language understanding and multimodal reasoning. | Essential for fine-tuning LLMs for Vietnamese NLP tasks. |
| **ViCode / ViCodeEval** | Vietnamese Code Generation | Generating code from Vietnamese prompts | Measures code generation accuracy and reasoning in Vietnamese context. | Important for developing Vietnamese coding assistants and educational tools. |
| **ViNLI / ViNLI-v2** | Vietnamese Natural Language Inference | Text Pair Classification (Entailment / Contradiction) | Measures logical entailment, semantic reasoning, and consistency in Vietnamese. | Key benchmark for evaluating NLI, semantic matching, and factual consistency. |
| **ViLegal-Bench / LegalSLM** | Vietnamese Legal Reasoning | Contract QA, Classification & Risk Extraction | Evaluates comprehension of Vietnamese law, legal precedent retrieval, and statutory interpretation. | Crucial domain benchmark for deploying enterprise and legal AI assistants in Vietnam. |
| **ViNMT / VLSP Translation** | Vietnamese Machine Translation | Text Translation (VN <-> EN / Regional Languages) | Measures translation quality, fluency, and tone preservation in domain-specific corpora. | Standard benchmark for multilingual LLM alignment and machine translation fine-tuning. |
| **ViNCS / Vi-CodeSwitch** | Code-Switching & Cross-Lingual | Conversational Mixed Text (VN-EN) | Assesses LLM stability and fluency when processing code-switched Vietnamese and English text. | Essential for real-world Vietnamese user chat applications and customer service bots. |

--- 

## Implementing fine-tuning with backbones:

- [x] Mistral backbones with SQUAD and ViQuAD for multilingual QA fine-tuning.

- [ ] LLaMA 3.1 with RAG and ViNLI for cross-lingual reasoning tasks.

- [ ] Qwen 3 series with MedQA and ViLegal-Bench for domain-specific medical and legal applications.