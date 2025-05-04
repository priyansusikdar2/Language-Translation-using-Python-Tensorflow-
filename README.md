# 🗣️ English-to-French Language Translator using TensorFlow (No NLP Libraries)

This project demonstrates a simple neural machine translation (NMT) system to translate English sentences into French using Python and TensorFlow — without using NLP-specific libraries.

## 📌 Features

- Sequence-to-sequence (Seq2Seq) model using TensorFlow and Keras.
- No external NLP libraries like NLTK, SpaCy, or HuggingFace.
- Uses basic tokenization and padding techniques.
- Trained on parallel English-French sentence pairs.

---

## 📁 Project Structure

```
language-translation/
├── data/
│   └── eng-fra.txt         # English-French sentence pairs (tab-separated)
├── model/
│   └── translator.h5       # Saved model (after training)
├── translator.py           # Main training and evaluation script
├── utils.py                # Tokenization, padding, and preprocessing functions
└── README.md               # Project documentation
```

---

## ⚙️ Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy

Install dependencies using:

```bash
pip install tensorflow numpy
```

---

## 🔄 How It Works

1. **Load Data**: Load sentence pairs from `eng-fra.txt`.
2. **Preprocess**:
   - Basic character-level or whitespace tokenization.
   - Map tokens to integer sequences.
   - Pad sequences for uniform length.
3. **Build Model**:
   - Encoder-Decoder architecture using LSTM/GRU.
   - Encoder processes input English sentence.
   - Decoder generates corresponding French translation.
4. **Train**:
   - Train the model on input/output pairs using teacher forcing.
5. **Translate**:
   - Provide an English sentence and decode the French translation step-by-step.

---

## 🚀 Example

```python
# Example usage
from translator import translate_sentence

print(translate_sentence("I love you"))
# Output: "Je t'aime"
```

---

## 🧠 Model Architecture

- **Encoder**: Embedding → LSTM
- **Decoder**: Embedding → LSTM → Dense
- Loss: SparseCategoricalCrossentropy
- Optimizer: Adam

---

## 📚 Dataset

Use a simple bilingual dataset like [ManyThings.org English-French pairs](https://www.manythings.org/anki/).

---

## 🛑 Limitations

- Only supports basic translation for short phrases.
- No attention mechanism (optional enhancement).
- No advanced tokenization or BPE (since NLP libs are not used).

---

## 📈 To Improve

- Add attention (Luong or Bahdanau).
- Use subword tokenization.
- Train on larger datasets.

---

## 🧑‍💻 Author

Made with 💻 and ☕ using Python & TensorFlow.

