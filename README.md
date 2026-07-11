## 🗺️ Roadmap & Implementation Plan

I am rebuilding this framework following a step-by-step curriculum to master LLM internals.

### 🏁 Phase 1: Core Architecture (The Brain)
- [ ] **Implement `nanochat/gpt.py`**: Build the Transformer architecture (Multi-Head Attention, MLP, LayerNorm).
- [ ] **Set up `nanochat/tokenizer.py`**: Implement a BPE (Byte Pair Encoding) wrapper compatible with GPT-4/Llama patterns.
- [ ] **Define Configs**: Adapt model hyperparameters from GPT-2 (124M) or Llama (Small variants).

### 📊 Phase 2: Data Engineering (The Fuel)
- [ ] **Build `nanochat/dataloader.py`**: Create a distributed, tokenized data loader to stream data shards efficiently.
- [ ] **Data Repackaging**: Write scripts to convert raw text (Fineweb/DCLM) into binary shards for high-speed training.

### 🚂 Phase 3: Pre-training (The Learning)
- [ ] **Develop `scripts/base_train.py`**: Write the main training loop with weight decay and learning rate scheduling.
- [ ] **Optimization**: Integrate `AdamW` and experiment with the `Muon` optimizer for faster convergence.
- [ ] **Validation**: Implement loss evaluation and basic sampling to see the model "learn" English.

### 💬 Phase 4: Alignment & Fine-tuning (The Chatbot)
- [ ] **SFT (Supervised Fine-Tuning)**: Train the base model on instruction datasets (e.g., `smoltalk`) using `scripts/chat_sft.py`.
- [ ] **Inference Engine**: Implement `nanochat/engine.py` with **KV Caching** for snappy, real-time response generation.
- [ ] **Chat UI**: Connect the model to the HTML/Web interface for a full ChatGPT-like experience.

### 🧪 Phase 5: Evaluation (The Exam)
- [ ] Run **MMLU** and **GSM8K** benchmarks to measure world knowledge and reasoning.
- [ ] Perform **HumanEval** to test Python coding capabilities.

## How to test Attention
```bash
uv run pytest -v test_attention.py
```