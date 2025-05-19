import tkinter as tk
from tkinter import messagebox
from nltk.corpus import wordnet as wn
import nltk

# Download resources (once)
nltk.download('wordnet')
nltk.download('omw-1.4')

# Translation function using WordNet
def translate_french_words():
    input_text = entry.get("1.0", tk.END).strip()
    if not input_text:
        messagebox.showwarning("Input Needed", "Please enter French words.")
        return

    french_words = [w.strip() for w in input_text.split(',') if w.strip()]
    results = {}

    for word in french_words:
        synsets = wn.synsets(word, lang='fra')
        translations = set()
        for syn in synsets:
            for lemma in syn.lemmas('eng'):
                translations.add(lemma.name().replace('_', ' '))
        results[word] = list(translations) if translations else ["No translation found"]

    # Display results
    output_text.delete("1.0", tk.END)
    for fr_word, en_list in results.items():
        output_text.insert(tk.END, f"{fr_word} → {', '.join(en_list)}\n")

# === GUI ===
root = tk.Tk()
root.title("French to English Translator (WordNet)")
root.geometry("500x400")

tk.Label(root, text="Enter French words (comma-separated):", font=("Arial", 12)).pack(pady=10)
entry = tk.Text(root, height=5, width=60)
entry.pack()

tk.Button(root, text="Translate", command=translate_french_words, bg="lightblue").pack(pady=10)

tk.Label(root, text="English Translations:", font=("Arial", 12)).pack()
output_text = tk.Text(root, height=10, width=60, bg="#f0f0f0")
output_text.pack(pady=5)

root.mainloop()
