
# 🧠 English to Italian Translation (Seq2Seq with Luong Attention)

This repository contains a PyTorch implementation of a **Sequence-to-Sequence model** for translating English sentences into Italian. The model uses **Luong Attention**, **beam search decoding**, and is evaluated using the **BLEU score**.

---

## 📂 Project Structure

```
.
├── notebook.ipynb           # Main training and evaluation notebook
├── data/                    # Preprocessed Tatoeba EN-IT data (or download at runtime)
├── models/                  # Saved model checkpoints (optional)
├── utils.py                 # Tokenization, vocabulary building, BLEU and beam search helpers
├── README.md                # This file
```

---

## 🧪 Model Summary

- **Architecture**: Seq2Seq (LSTM encoder-decoder)  
- **Attention**: Luong (general scoring)  
- **Decoder Input**: Teacher forcing with scheduled decay  
- **Embedding**: Learned from scratch  
- **Evaluation**: BLEU score on 100 validation samples  
- **Decoding**: Beam Search (`beam_width=5`)

---

## 📊 Results

| Metric        | Score     |
|---------------|-----------|
| BLEU (val set) | **0.63** |
| Best Epoch     | ~5 (early stopping monitored on val loss) |

### 🔍 Sample Translations:
```
EN: Hello, how are you?
IT: in mare che avete visto

EN: I love machine learning.
IT: io sono agitato

EN: This book is very interesting.
IT: tutti sono impedito
```

---

## 📦 Requirements

- Python ≥ 3.8  
- PyTorch ≥ 1.11  
- HuggingFace `datasets`  
- `nltk`, `tqdm`, `sklearn`

Install with:
```bash
pip install torch datasets nltk tqdm scikit-learn
```

---

## 🚀 Training

To train the model:
```python
# From notebook
train(model, train_loader, optimizer, criterion, ...)
```

The notebook includes:
- Scheduled teacher forcing  
- Validation loss tracking  
- Early stopping and checkpointing

---

## 📈 Evaluation

After training, the model is evaluated using:
- Token-level accuracy (excluding `<PAD>`)
- BLEU score (computed via NLTK)
- Beam search predictions

```python
bleu_score = compute_bleu(model, val_loader, ...)
print(f"BLEU: {bleu_score * 100:.2f}")
```

---

## 🔮 Future Work

- Use pretrained embeddings (e.g., GloVe, fastText)  
- Add positional encoding or Transformer decoder  
- Improve generalization with regularization  
- Evaluate on full test set and larger corpora

---

## 📝 License

This project is provided for educational purposes only.
