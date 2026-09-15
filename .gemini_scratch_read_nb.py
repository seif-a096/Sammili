import torch
import matplotlib.pyplot as plt
import sys

sys.stdout.reconfigure(encoding='utf-8')

# load names
with open("names.txt", "r", encoding="utf-8") as f:
    names = [line.strip() for line in f if line.strip()]

# add boundary tokens
names = [['.']+list(n)+['.'] for n in names]

# build bigram counts
seq = {}
for name in names:
    for ch1, ch2 in zip(name, name[1:]):
        seq[(ch1, ch2)] = seq.get((ch1, ch2), 0) + 1

# char dicts
flat = [ch for name in names for ch in name]
ch_dict = {ch: i for i, ch in enumerate(sorted(set(flat)))}
ch_dict_reversed = {i: ch for ch, i in ch_dict.items()}
n = len(ch_dict)

# fill array
arr = torch.zeros((n, n), dtype=torch.int32)
for (ch1, ch2), count in seq.items():
    arr[ch_dict[ch1], ch_dict[ch2]] = count

# plot
plt.figure(figsize=(16, 16))
plt.imshow(arr, cmap="Blues")

for i in range(n):
    for j in range(n):
        bigram = ch_dict_reversed[i] + ch_dict_reversed[j]
        count = arr[i, j].item()
        plt.text(j, i, bigram, ha="center", va="bottom", color="gray", fontsize=7)
        plt.text(j, i, count, ha="center", va="top", color="gray", fontsize=6)

plt.axis("off")
plt.tight_layout()
plt.savefig("outputs/bigram_matrix.png", dpi=150, bbox_inches="tight", facecolor="white")
print("Saved to outputs/bigram_matrix.png")
