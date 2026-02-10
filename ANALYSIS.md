# Forward-Forward + SSM: MVP Analysis

*Research notes for local AI training without backpropagation*

---

## 1. Forward-Forward Algorithm Summary (from Hinton's paper)

### Core Idea
Replace backprop's forward+backward passes with **two forward passes**:
- **Positive pass**: Real data → maximize "goodness" 
- **Negative pass**: Fake/corrupted data → minimize "goodness"

### Goodness Function
```
goodness = Σ(y_j²)  # sum of squared activations
p(positive) = σ(goodness - θ)  # logistic of goodness minus threshold
```

### Key Properties
1. **Layer-wise learning** — each layer has its own objective, no need to backprop through whole network
2. **Layer normalization** — removes length, passes only orientation to next layer
3. **Incremental** — can learn while streaming data, no need to store activations
4. **CPU-friendly** — designed for analog hardware, low power
5. **Black-box compatible** — works even with unknown non-linearities

### Supervised Learning Approach
Include label in input. Positive = correct label, Negative = wrong label.
```python
# Overlay label on first 10 pixels
x_pos = overlay_label(image, correct_label)
x_neg = overlay_label(image, wrong_label)
```

### Performance
- MNIST: **1.36% error** (vs 1.4% for standard backprop)
- Takes ~60 epochs (vs ~20 for backprop) — slower but works!
- With jittering: 0.64% error (matches ConvNets)

---

## 2. Available Implementations

### A. mpezeshki/pytorch_forward_forward ⭐ (Best starting point)
- **Stars:** ~800
- **Code:** 150 lines, pure PyTorch
- **Task:** MNIST classification
- **Hardware:** CUDA (but easy to modify for CPU)
- **Status:** Clean, minimal, well-documented
- **Location:** `/root/clawd/research/ff-experiment/ff-original/`

### B. cozheyuanzhangde/Forward-Forward
- **Stars:** ~10
- **Features:** Claims "Language Modeling" in progress
- **Status:** Work in progress, but has batch training modes
- **Location:** `/root/clawd/research/ff-experiment/ff-language/`

### C. Ads97/ForwardForward
- **Stars:** ~33
- **Features:** Sentiment analysis on text! 
- **Paper:** https://arxiv.org/abs/2307.04205
- **Status:** Extended FF to text classification

---

## 3. Recommended Datasets

### For Initial Experiments (Small)

| Dataset | Size | Description | Why Good |
|---------|------|-------------|----------|
| **TinyShakespeare** | 1.1MB | All Shakespeare | Classic, char-level works |
| **Penn Treebank** | ~5MB | WSJ articles | Standard LM benchmark |
| **WikiText-2** | ~12MB | Wikipedia | Clean, well-structured |

### For Scaling Up

| Dataset | Size | Description |
|---------|------|-------------|
| **TinyStories** | 2.2B tokens | Simple children's stories (Microsoft) |
| **OpenWebText** | 38GB | Web text (GPT-2 training data) |
| **The Pile** | 800GB | Diverse text corpus |

### Recommendation: Start with TinyShakespeare
```bash
# Download
curl -o /root/clawd/research/datasets/shakespeare.txt \
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```
- Small enough for CPU experiments
- Character-level = simple tokenization
- Famous = easy to verify if it's learning ("To be or not to be...")

---

## 4. MVP Plan

### Phase 1: Validate FF on CPU (Week 1)
```
Goal: Prove FF works without GPU

1. Modify mpezeshki's code to run on CPU
2. Train on MNIST (should get ~1.4% error)
3. Measure: RAM, time per epoch, convergence
4. Document baseline performance
```

### Phase 2: FF for Text (Week 2)
```
Goal: Adapt FF for character/token prediction

1. Build character-level tokenizer for Shakespeare
2. Create positive/negative samples:
   - Positive: real next-char prediction
   - Negative: wrong next-char prediction
3. Train FF layers on sequence prediction
4. Measure: perplexity, generation quality
```

### Phase 3: Tiny SSM + FF (Week 3-4)
```
Goal: Replace MLP layers with SSM layers

1. Implement minimal S4/Mamba layer (~100 lines)
2. Replace FFLayer's linear transform with SSM
3. Keep FF's goodness-based training
4. Compare: memory, speed, quality
```

### Phase 4: Engram Integration (Week 5+)
```
Goal: Learn from our conversation logs

1. Feed Engram JSONL as training data
2. Test incremental learning (add new data daily)
3. Query: "Who is Victor?" after training
4. Measure: retention, forgetting, accuracy
```

---

## 5. Key Questions to Answer

1. **Does FF work on CPU with 16GB RAM?**
   - Hypothesis: Yes, it's designed for low-power hardware

2. **Can FF learn text/sequences?**
   - Ads97 shows sentiment works, but generative LM is harder

3. **Can we replace linear layers with SSM?**
   - Novel research territory — our contribution

4. **Does FF enable incremental learning?**
   - Key differentiator from backprop

---

## 6. First Experiment: Run Today

```bash
cd /root/clawd/research/ff-experiment/ff-original

# Modify to use CPU
sed -i 's/.cuda()//g' main.py
sed -i 's/x.cuda()/x/g' main.py
sed -i 's/y.cuda()/y/g' main.py

# Run
python main.py
```

This will:
- Download MNIST
- Train 2-layer FF network
- Report train/test error
- Prove the concept works locally

---

## 7. Why This Matters

From Hinton's paper:

> "If you want your trillion parameter neural net to only consume a few watts, 
> mortal computation may be the only option."

We're not building a trillion parameters. We're building something that can:
- **Learn on CPU** (no cloud dependency)
- **Learn incrementally** (from Engram, daily)
- **Run locally** (on a $35 Raspberry Pi eventually)
- **Evolve** (the caterpillar designs the butterfly)

"El hambre agudiza el ingenio."

---

*Created: 2026-02-10*
*Authors: VictorIA + Victor*
