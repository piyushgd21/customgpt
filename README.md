# Custom Character-Level GPT on Case Files

This project implements a **lightweight GPT-style language model** trained from scratch on intelligence-related case files. The model generates predictions such as:

> The next bombing location might be...

Built entirely in PyTorch, this model captures domain-specific patterns using a compact, character-level transformer.

<img width="1436" height="342" alt="image" src="https://github.com/user-attachments/assets/2d919e09-b634-4f8a-85ac-9e1c53764fbd" />

---

## 🚀 Features

- Character-level GPT trained from scratch on `.txt` case files
- Custom vocabulary mappings stored in `meta.pkl` (`stoi` and `itos`)
- Lightweight transformer (6 layers · 6 heads · 128 embedding size)
- Sampling with configurable temperature and top-k
- CPU-compatible (no GPU required)

---

## 📁 Directory Structure

<img width="459" height="248" alt="image" src="https://github.com/user-attachments/assets/7ca33b80-3ef6-4ebd-a037-413d4e78fa38" />

## ⚙️ Setup

```bash
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

## Step 2: Train the Model

Ensure your training config is set in `config/train_casefiles_char.py`, then run:

```bash
python train.py config/train_casefiles_char.py
```

Training will save checkpoints to `out-casefiles-char/`.

---

## Step 3: Generate Predictions

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

## Architecture Summary

* Character-level transformer
* Positional embeddings + token embeddings
* Multi-head causal self-attention
* FeedForward MLP (with GELU activation)
* Output projection to vocab size + softmax sampling

---

## Technologies Used

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
