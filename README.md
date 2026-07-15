# Neural Networks: Zero to Hero — Study & Implementation

A from-scratch, deeply-annotated walkthrough of Andrej Karpathy's
[**Neural Networks: Zero to Hero**](https://github.com/karpathy/nn-zero-to-hero) series.

This is my learning log. Instead of copying the reference code, I **re-implement each
component by hand**, annotate *why* every line works, verify my gradients against PyTorch,
and solve the exercises. The goal is a first-principles understanding of how neural networks
— from automatic differentiation up to language models — actually work under the hood.

## What's inside

### `src/micrograd/` — a tiny autograd engine
- **`micrograd.ipynb`** — A `Value` class that builds a computational graph during the forward
  pass, with **reverse-mode backprop** via topological sort and correct gradient *accumulation*
  (a bug I hit and fixed — see commit history). Built up into a `Neuron → Layer → MLP` and a full
  **training loop** (forward → zero-grad → backward → SGD), with gradients **cross-checked against PyTorch**.
- **`micrograd_exercises.ipynb`** — Analytical, numerical, and symmetric-derivative gradients;
  extending `Value` with **softmax + negative log-likelihood** (`exp`/`log`/`div` from scratch),
  every gradient **verified against `torch`**.

### `src/makemore/` — a character-level language model
Built up over five parts, each with worked exercises (`exNN.ipynb`):

- **`part1_bigram/`** — The *same* bigram model built two ways:
  - **Statistical**: a 27×27 count matrix normalized into distributions, sampled with
    `torch.multinomial`, with Laplace smoothing.
  - **Neural net**: the same problem as one linear layer — one-hot inputs, logits as **log-counts**,
    **softmax** derived from first principles, **negative log-likelihood** loss derived step by step,
    and **gradient descent** — showing L2 regularization *is* the smoothing from the statistical model.
  - Annotated throughout on **broadcasting**, tensor shapes, and what each `@` computes.
- **`part2_MLP/`** — The Bengio-style **MLP** language model: a character **embedding table**,
  a hidden layer, and a softmax output, with a train/dev/test split and a learning-rate sweep.
- **`part3_MLP2/`** — The same MLP made to **train well**: careful weight/gain initialization,
  **BatchNorm**, and activation/gradient **diagnostics**. Exercises fold BatchNorm into the
  preceding `Linear` at inference and connect the `Linear` gradients to Nielsen's BP1–BP4.
- **`part4_backprop/`** — **Manual backpropagation** through the entire MLP — every intermediate
  gradient (cross-entropy, BatchNorm, matmuls, the embedding lookup) derived by hand and
  **cross-checked against PyTorch autograd**. `doc/dhprebn_derivation.md` works out the one-shot
  BatchNorm gradient (the math behind Exercise 2).
- **`part5_WavNet/`** — A hierarchical, **WaveNet**-style model that fuses characters in stages
  via a custom `FlattenConsecutive` layer, built on a reusable PyTorch-like module API
  (`Linear`, `BatchNorm1d`, `Tanh`, `Embedding`, `Sequential`) — making the network *deeper* and
  *tree-structured* rather than wider.

### `src/gpt/` — building a GPT from scratch
A decoder-only **Transformer** trained on the Tiny Shakespeare corpus (`data/input.txt`), built up from
a bigram baseline to the full model.

- **`gpt-dev.ipynb`** — The working notebook: develops the **self-attention** mechanism from first
  principles (the averaging-as-matrix-multiply trick → masked, scaled dot-product attention).
- **`bigram.py`** — The starting-point baseline: a token-embedding bigram language model with a
  training loop and sampling.
- **`v2.py`** — The full **GPT**: token + position embeddings, **multi-head self-attention**, a
  **feed-forward** layer, stacked **Transformer blocks** with **residual connections** and
  **LayerNorm**, plus **dropout** — scaled up (`n_embd=384`, 6 heads, 6 layers, `block_size=256`)
  to generate Shakespeare-like text.

### `src/token/` — building a BPE tokenizer (minBPE)
- **`dev.ipynb`** — The byte-pair encoding algorithm from first principles: counting adjacent
  byte-pair frequencies, iteratively merging the most common pair into a new token, and building
  up the `encode`/`decode` machinery on top of the learned merges.

### `src/nanogpt/` — reproducing GPT-2 (124M) *(in progress)*
Following the *Let's reproduce GPT-2* lecture, rebuilding the GPT-2 architecture in raw PyTorch
and loading the real OpenAI weights to validate it.

- **`train_gpt2.py`** — The GPT-2 `nn.Module` from scratch: `CausalSelfAttention` (batched
  multi-head attention via a single fused `c_attn` projection), the `MLP`/feed-forward block with
  `tanh`-approximate GELU, pre-norm `Block`s, and the full `GPT` model (token + position
  embeddings, stacked blocks, final LayerNorm, LM head). `from_pretrained` loads the HuggingFace
  `GPT2LMHeadModel` checkpoint into this module, transposing the `Conv1D`-style weights
  (`c_attn`, `c_proj`, `c_fc`) to match `nn.Linear`'s convention. Currently wiring up the forward
  pass to produce logits from the loaded weights.
- **`play.ipynb`** — Scratch notebook for exploring the GPT-2 (124M) checkpoint.

## Key concepts

Reverse-mode autodiff · computational graphs · analytical vs. numerical gradients ·
softmax & cross-entropy derived by hand · verifying custom gradients against PyTorch ·
maximum-likelihood estimation · a statistical model recast as an equivalent neural net ·
broadcasting & matrix shapes · regularization as smoothing · character embeddings & the MLP
language model · weight initialization · BatchNorm · activation/gradient diagnostics ·
manual backprop through a full network · hierarchical/WaveNet-style models · self-attention &
scaled dot-product attention · multi-head attention · Transformer blocks · residual connections ·
LayerNorm · dropout · decoder-only GPT language model · byte-pair encoding (BPE) tokenization ·
loading pretrained HuggingFace weights into a from-scratch module · GPT-2 architecture reproduction.

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt    # numpy, matplotlib, graphviz, torch
jupyter notebook                   # open any .ipynb and run top-to-bottom
```
> Graph visualizations need the Graphviz binary (`brew install graphviz`).

## Progress

- [x] **micrograd** — autograd engine, MLP, training loop
- [x] **micrograd exercises** — gradients & softmax, verified against PyTorch
- [x] **makemore (bigram)** — statistical + single-layer neural model
- [x] **makemore (MLP)** — Bengio-style embedding + hidden-layer model
- [x] **makemore (MLP2)** — initialization, BatchNorm, diagnostics
- [x] **makemore (backprop)** — manual backprop through the whole net
- [x] **makemore (WaveNet)** — hierarchical model with `FlattenConsecutive`
- [x] **Building GPT** — self-attention and the Transformer from scratch
- [x] **Tokenization (minBPE)** — byte-pair encoding from scratch
- [ ] **Reproducing GPT-2 (nanoGPT)** — architecture in progress, loading pretrained weights

---

All exercises and the lecture series are by
**[Andrej Karpathy](https://github.com/karpathy/nn-zero-to-hero)**; the implementations,
notes, and solutions here are my own.
