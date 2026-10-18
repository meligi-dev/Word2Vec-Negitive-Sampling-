# Word2Vec: Skip-Gram with Negative Sampling

This repository implements the **Word2Vec (Skip-Gram)** architecture from scratch in PyTorch, with a primary focus on **Negative Sampling (NEG)** as introduced by Mikolov et al. (Google, 2013). It contrasts the naive, computationally prohibitive full-softmax Skip-Gram model with the scalable negative sampling formulation and provides 2D geometric projections and cosine similarity visualizations of learned word representations.

---

## 📚 Notebooks in this Directory

1. **[Word2Vec(Negitive_Sampling).ipynb](file:///d:/Learn_NLP/Word2Vec/Word2Vec(Negitive_Sampling).ipynb)**: Full from-scratch implementation of Skip-Gram with Negative Sampling, featuring custom unigram table generation ($U(w)^{3/4}$), noise sampling, binary logistic loss, and 2D embedding space & similarity heatmap visualization.
2. **[Word2Vec(SkipGram).ipynb](file:///d:/Learn_NLP/Word2Vec/Word2Vec(SkipGram).ipynb)**: Reference implementation of the baseline Skip-Gram model with standard Softmax over the entire vocabulary $V$.

---

## 🧠 Theoretical Foundations

### 1. The Skip-Gram Objective & The Softmax Bottleneck
The standard Skip-Gram model aims to maximize the probability of context (outside) words $w_O$ given a center word $w_C$:

$$P(w_O \mid w_C) = \frac{\exp({v_{w_O}'}^\top v_{w_C})}{\sum_{w \in V} \exp({v_w'}^\top v_{w_C})}$$

* **The Problem**: Computing the denominator requires calculating dot products and exponentiations across the **entire vocabulary $V$** at every training step. For real-world vocabularies ($|V| \approx 10^5 - 10^7$), this gradient computation is prohibitively expensive ($O(|V|)$ per word).

---

### 2. Negative Sampling (NEG) Formulation
Mikolov et al. (2013) recast the multi-class categorization problem into a collection of independent **binary logistic regressions**:
* Differentiate the true outside word $w_O$ (positive label $y=1$) from $K$ randomly sampled noise words $w_{N_1}, \dots, w_{N_K}$ drawn from a noise distribution $P_n(w)$ (negative label $y=0$).

The Negative Sampling objective for a center-outside pair $(w_C, w_O)$ is:

$$J_{\text{NEG}}(\theta) = -\log \sigma({u_o}^\top v_c) - \sum_{k=1}^K \log \sigma(-{u_{n_k}}^\top v_c)$$

Where:
* $\sigma(x) = \frac{1}{1 + e^{-x}}$ is the sigmoid activation function.
* $v_c$: Embedding vector of center word $w_c$ (`embedding_center`).
* $u_o$: Embedding vector of positive context word $w_o$ (`embedding_outside`).
* $u_{n_k}$: Embedding vector of the $k$-th negative sample (`embedding_outside`).
* $K$: Number of negative samples (typically $5 - 20$ for small datasets, $2 - 5$ for large corpora).

```
[ Center Word: v_c ]
        |
        +---- Dot Product with Positive Context (u_o) -----> log-sigmoid( u_o^T v_c )    --> Maximize
        |
        +---- Dot Product with Neg Sample 1 (u_n1)   -----> log-sigmoid( -u_n1^T v_c )  --> Maximize
        |
        +---- Dot Product with Neg Sample 2 (u_n2)   -----> log-sigmoid( -u_n2^T v_c )  --> Maximize
        |
        +---- Dot Product with Neg Sample K (u_nK)   -----> log-sigmoid( -u_nK^T v_c )  --> Maximize
```

---

### 3. Noise Distribution: The 3/4 Unigram Power Law
Negative words are sampled from a smoothed unigram distribution:

$$P_n(w) = \frac{U(w)^{3/4}}{\sum_{w' \in V} U(w')^{3/4}}$$

* **Why $3/4$?**: Raising the unigram frequency $U(w)$ to the power of $0.75$ significantly elevates the sampling probability of rare words while dampening the dominance of ultra-frequent stop words (*"the"*, *"is"*, *"a"*).
* **Table Lookup**: Implemented via a discrete unigram table where each word $w$ appears $\lfloor (U(w)^{0.75}) / z \rfloor$ times ($z = 0.001$), enabling $O(1)$ random choice sampling.

---

## 🔬 Implementation Walkthrough

### 1. Data Preparation & Sliding Window
Given a sample corpus:
```python
corpus = [
    "apple banana fruit", "banana apple fruit", "banana fruit apple",
    "dog cat animal", "cat animal dog", "cat dog animal",
    "grape apple apple", "fish dog dog"
]
```
* **Vocabulary Extraction**: Generates bi-directional lookup mappings (`word2indx` and `indx2word`) with `<UNK>` fallback.
* **Skip-Gram Pairs**: A sliding context window extracts `(center, outside)` training pairs:
  - e.g., `["apple", "banana"]`, `["apple", "fruit"]`, `["dog", "cat"]`.

### 2. Unigram Table & Negative Sampler
* Constructs the $U(w)^{0.75}$ frequency table.
* For each positive pair $(w_c, w_o)$, draws $K = 5$ negative tokens from the unigram table, ensuring $w_n \neq w_o$.

### 3. Model Architecture (`skipgramNeg`)
Implemented directly in PyTorch:
```python
class skipgramNeg(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super(skipgramNeg, self).__init__()
        self.embedding_center = nn.Embedding(vocab_size, emb_size)
        self.embedding_outside = nn.Embedding(vocab_size, emb_size)
        self.logsigmoid = nn.LogSigmoid()

    def forward(self, center, outside, negative_words):
        # Center: [batch_size, 1]
        # Outside: [batch_size, 1]
        # Negative: [batch_size, K]
        center_emb = self.embedding_center(center)
        outside_emb = self.embedding_outside(outside)
        neg_emb = self.embedding_outside(negative_words)

        # Positive term: log(sigma(u_o^T v_c))
        uocv = outside_emb.bmm(center_emb.transpose(1, 2)).squeeze(2)
        pos_loss = self.logsigmoid(uocv)

        # Negative term: sum_k log(sigma(-u_nk^T v_c))
        ukvc = -neg_emb.bmm(center_emb.transpose(1, 2)).squeeze(2)
        neg_loss = self.logsigmoid(torch.sum(ukvc, 1, keepdim=True))

        loss = pos_loss + neg_loss
        return -torch.mean(loss)
```

---

## 📈 Training & Geometric Results

* **Embedding Dimension**: `2` (enables direct 2D Cartesian inspection without PCA distortion).
* **Batch Size**: `2`
* **Optimizer**: `Adam(lr=0.001)`
* **Epochs**: `5,000`

### Convergence Progression:
* **Epoch 1,000**: Loss `0.0743`
* **Epoch 2,000**: Loss `0.5186`
* **Epoch 4,000**: Loss `0.0611`
* **Epoch 5,000**: Loss `0.6322`

---

## 🎨 Learned Embedding Visualizations

The trained 2-dimensional embeddings demonstrate distinct semantic cluster separation:

### 1. 2D Coordinate Space
* **Fruit Cluster**: `apple`, `banana`, `fruit`, `grape` occupy a shared localized region in vector space.
* **Animal Cluster**: `dog`, `cat`, `animal`, `fish` form a distinct, well-separated cluster.

### 2. Pairwise Cosine Similarity Heatmap
Normalizing vectors ($\frac{v}{\|v\|_2}$) and calculating $S_{ij} = v_i \cdot v_j^\top$:
* Strongly positive cosine similarities ($\approx +0.8$ to $+1.0$) between words within the same category (e.g., `apple` $\leftrightarrow$ `banana`, `cat` $\leftrightarrow$ `dog`).
* Orthogonal or negative cosine similarities ($\le 0.0$) between cross-category words (e.g., `banana` $\leftrightarrow$ `dog`).

---

## 📁 Repository Structure

```
Word2Vec/
├── Word2Vec(Negitive_Sampling).ipynb  # Core notebook: Negative Sampling formulation & visualizations
├── Word2Vec(SkipGram).ipynb          # Baseline notebook: Full Softmax Skip-Gram implementation
├── README.md                          # Comprehensive documentation, equations, and insights
└── .gitignore                         # Ignores Jupyter checkpoints and Python caches
```

---

## 🚀 Setup & Execution

### 1. Requirements
Install the necessary packages:
```bash
pip install numpy torch matplotlib
```

### 2. Run the Notebooks
Launch Jupyter Notebook:
```bash
# Run Negative Sampling implementation
jupyter notebook "Word2Vec/Word2Vec(Negitive_Sampling).ipynb"

# Run Full Softmax baseline
jupyter notebook "Word2Vec/Word2Vec(SkipGram).ipynb"
```

---

## 📚 References
* Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013). *Distributed representations of words and phrases and their compositionality*. Advances in Neural Information Processing Systems (NeurIPS 2013), 3111–3119.
