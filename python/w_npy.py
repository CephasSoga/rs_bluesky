

import json
import numpy as np

with open(r"C:\Users\cepha\OneDrive\Bureau\Cube\gaello v.2\rs_bluesky\naive_bayes_model.json") as f:
    model = json.load(f)

class_probs = model["class_probabilities"]
word_probs  = model["word_probabilities"]

# Order the classes consistently
labels = list(class_probs.keys())

# Collect the full vocabulary
vocab = sorted({w for probs in word_probs.values() for w in probs.keys()})
vocab_index = {w: i for i, w in enumerate(vocab)}

# Build W matrix: (num_classes, vocab_size)
W = np.full((len(labels), len(vocab)), fill_value=-1e9, dtype=np.float32)  # log(0) ~ -inf

for ci, label in enumerate(labels):
    probs = word_probs.get(label, {})
    for w, p in probs.items():
        if p > 0:
            W[ci, vocab_index[w]] = np.log(p)

# Save
np.save("w.npy", W)

with open("labels.txt", "w") as f:
    for l in labels:
        f.write(l + "\n")

with open("vocab.txt", "w") as f:
    for w in vocab:
        f.write(w + "\n")

with open("vocab.json", "w") as f:
    json.dump(vocab_index, f, indent=2)
