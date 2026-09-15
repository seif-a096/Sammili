# 🔤 Sammili - Character-Level Name Generation

> **From brute-force counting to gradient descent — learning to generate Arabic names one character at a time.**

[![Python](https://img.shields.io/badge/Python-3.14-blue?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14-EE4C2C?logo=pytorch)](https://pytorch.org)
[![Dataset](https://img.shields.io/badge/🤗_Dataset-Egyptian_Names-FFD21E)](https://huggingface.co/datasets/Abdullah-afify/egyptian-names)

A **bigram character-level language model** trained on **44,626 Egyptian Arabic names**. Two approaches built side-by-side — a pure counting model (the perfect brute-force baseline) and a single-layer neural network that learns to mimic it through gradient descent.

_Can a neural network with a single weight matrix recover the same probability table that we can compute exactly by counting?_

### Here's what they generate (same seed, 20 samples):

| #   | 📊 Counting Model       | 🧠 Neural Network       |
| --- | ----------------------- | ----------------------- |
| 1   | تاثابولح                | الثابولح                |
| 2   | ادي                     | ادي                     |
| 3   | ه                       | اتيان                   |
| 4   | المراميالش              | اميالش                  |
| 5   | الهامبقرالجايه          | ابهامبقرالجايه          |
| 6   | مجوعكال                 | مجوعكال                 |
| 7   | ي                       | اوفناك                  |
| 8   | فناك                    | مربنسوينى               |
| 9   | مربنسوينى               | ازت                     |
| 10  | ازت                     | اح                      |
| 11  | ن                       | نيمابي                  |
| 12  | منيمالشيهري             | هري                     |
| 13  | ا                       | ا                       |
| 14  | سقرقاليتاصي             | سقرقانيتاصي             |
| 15  | ايوميان                 | ايوميان                 |
| 16  | معبوضي                  | معبوضي                  |
| 17  | عبتلامسمح               | التلامسمح               |
| 18  | الحميزقصفيعتاوجيوقللياه | الحميزقصفيعتاوجيوقللياه |
| 19  | عبويفالعضبوتحالبودخ     | عبويفاسيضبوتحان         |
| 20  | ان                      | الخ                     |

> Both produce similar gibberish — _exactly right_ for a bigram model. With only one character of context, there's no way to learn real name structure. That's what n-grams and transformers are for.

---

## 📦 The Data

**44,626 Egyptian Arabic names** from [Abdullah-afify/egyptian-names](https://huggingface.co/datasets/Abdullah-afify/egyptian-names) on 🤗 HuggingFace.

```python
data = load_dataset("Abdullah-afify/egyptian-names")
names = list(data["train"]["name_ar"])  # ['محمد', 'احمد', 'علي', 'محمود', ...]
```

Each name is wrapped with a `.` boundary token to mark start/end:

```python
# '.محمد.' → the model learns:
#   . → م  (what characters start names?)
#   م → ح  (what follows م?)
#   ح → م  (what follows ح?)
#   م → د  (what follows م in this context?)
#   د → .  (what characters end names?)
```

| Stat              | Value                |
| ----------------- | -------------------- |
| Total names       | 44,626               |
| Total bigrams     | 303,362              |
| Unique characters | 35 (34 Arabic + `.`) |

After normalization (hamza variants → `ا`, taa marbuta → `ه`), the alphabet collapses from 38 to 34 unique Arabic characters.

---

## 📊 Model 1 — Counting (Brute Force)

The _perfect_ baseline. No learning, no gradients — just count every character pair and normalize.

```python
seq = {}
for name in names:
    for ch1, ch2 in zip(name, name[1:]):
        seq[(ch1, ch2)] = seq.get((ch1, ch2), 0) + 1

# ('.', 'ا'): 17214   ← most names start with alef
# ('ي', '.'): 9468    ← many names end with ya
# ('ا', 'ل'): 15568   ← الـ (al-) is extremely common
```

Store in a 35×35 tensor and normalize with Laplace smoothing:

```python
arr = torch.zeros((35, 35), dtype=torch.int32)
for (ch1, ch2), count in seq.items():
    arr[ch_dict[ch1], ch_dict[ch2]] = count

P = (counts + 1) / (counts.sum(dim=1, keepdim=True) + 35)
```

The `+1` smoothing ensures every bigram has nonzero probability — no `log(0)` during evaluation.

**Baseline loss: `2.5870` average negative log-likelihood**

---

## 🧠 Model 2 — Neural Network

Can a neural network _learn_ that same probability table from scratch? Yes — with a single matrix multiply and softmax.

```
Input (one-hot, 35) → W (35×35, learnable) → Softmax → Output (35 probabilities)
```

No hidden layers, no biases, no activation functions. The weight matrix `W` _is_ the model — each row learns the log-probability distribution for one input character.

### Forward Pass

```python
x_enc = F.one_hot(torch.tensor(xs), num_classes=35).float()
logits = x_enc @ W           # (N, 35) — raw scores
counts = logits.exp()         # exponentiate to get positive "counts"
probs = counts / counts.sum(dim=1, keepdim=True)  # normalize → probabilities
```

### Loss: NLL + L2 Regularization

```python
loss = -torch.log(probs[torch.arange(len(ys)), ys]).mean() + 0.01 * (W**2).mean()
```

The `0.01 * (W²)` term is L2 regularization — penalizes large weights, pushing toward uniform distributions when data is scarce. Acts exactly like Laplace smoothing in the counting model.

### Training

```python
for epoch in range(500):
    logits = x_enc @ W
    probs = logits.exp() / logits.exp().sum(dim=1, keepdim=True)
    loss = -torch.log(probs[torch.arange(len(ys)), ys]).mean() + 0.01 * (W**2).mean()
    loss.backward()
    with torch.no_grad():
        W -= 50 * W.grad
        W.grad.zero_()
```

```
Epoch   0   Loss: 4.0612
Epoch  50   Loss: 2.6653
Epoch 100   Loss: 2.6337
Epoch 300   Loss: 2.6156
Epoch 490   Loss: 2.6130
```

The NN converges to **`2.6130`** — very close to the counting model's **`2.5870`**. The small gap is the L2 regularization cost; the network has effectively _recovered_ the bigram statistics from data alone.

---

## 🚀 How to Run

```bash
git clone https://github.com/seif-a096/Sammili.git
cd Sammili

uv sync
uv run jupyter notebook notebook.ipynb
```

| Package      | Purpose                             |
| ------------ | ----------------------------------- |
| `torch`      | Tensor ops, autograd, training loop |
| `datasets`   | Loading the HuggingFace dataset     |
| `matplotlib` | Bigram heatmap visualization        |
| `ipykernel`  | Jupyter notebook support            |

---

## 📁 Repository Structure

```
Sammili/
├ README.md              you are here
├ notebook.ipynb          the full interactive walkthrough
├ names.txt              44,626 normalized Arabic names
├ main.py                entry point stub
├ pyproject.toml          project config & dependencies
└ uv.lock               lockfile
```

---

<p align="center">
  <i>سمّلي — from counting to calculus, one character at a time.</i>
</p>
