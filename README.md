# 🧠 Custom Character-Level GPT on Case Files

This project implements a **lightweight GPT-style language model** trained from scratch on intelligence-related case files. The model generates predictions such as:

> The next bombing location might be...

Built entirely in PyTorch, this model captures domain-specific patterns using a compact, character-level transformer.

---

## 🚀 Features

- Character-level GPT trained from scratch on `.txt` case files
- Custom vocabulary mappings stored in `meta.pkl` (`stoi` and `itos`)
- Lightweight transformer (6 layers · 6 heads · 128 embedding size)
- Sampling with configurable temperature and top-k
- CPU-compatible (no GPU required)

---

## 📁 Directory Structure


data/
└── casefiles/
├── input.txt         # Combined case file input
├── train.bin         # Training token IDs
├── val.bin           # Validation token IDs
└── meta.pkl          # Vocabulary mappings

config/
└── train\_casefiles\_char.py  # Training configuration

out-casefiles-char/
└── ckpt.pt              # Model checkpoint (auto-generated)


## ⚙️ Setup

```bash
# Clone the repository
git clone https://github.com/your-username/custom-gpt-casefiles.git
cd custom-gpt-casefiles

# Install dependencies
pip install torch numpy tiktoken
````

---

## 🔧 Step 1: Prepare Dataset

Merge your `.txt` files into one:

```bash
cat path/to/files/*.txt > data/casefiles/input.txt
```

Run the tokenizer to prepare `train.bin`, `val.bin`, and `meta.pkl`:

```bash
python data/shakespeare_char/prepare.py
```

---

## 🏋️ Step 2: Train the Model

Ensure your training config is set in `config/train_casefiles_char.py`, then run:

```bash
python train.py config/train_casefiles_char.py
```

Training will save checkpoints to `out-casefiles-char/`.

---

## 🔮 Step 3: Generate Predictions

```bash
python sample.py \
  --out_dir=out-casefiles-char \
  --start="The next bombing location might be" \
  --temperature=0.8 \
  --top_k=50 \
  --device=cpu
```

You’ll see multiple plausible paragraph completions based on your domain.

---

## 🧠 Architecture Summary

* Character-level transformer
* Positional embeddings + token embeddings
* Multi-head causal self-attention
* FeedForward MLP (with GELU activation)
* Output projection to vocab size + softmax sampling

---

## 📦 Technologies Used

* Python
* PyTorch
* Custom transformer architecture
* CLI-based training & sampling interface

---

## 📌 Notes

* This is a **domain-specific LLM** trained from scratch (not a fine-tune).
* Ideal for learning, prototyping, or custom text generation tasks.
* Trains and samples efficiently on CPU.

---

## ✍️ Author

**Piyush Deshpande**
