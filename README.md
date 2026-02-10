# Forward-Forward Algorithm - Extended

> Fork of [mpezeshki/pytorch_forward_forward](https://github.com/mpezeshki/pytorch_forward_forward) with improvements and experiments.

## What is Forward-Forward?

An alternative to backpropagation proposed by Geoffrey Hinton. Instead of forward + backward passes, it uses **two forward passes**:
- **Positive pass**: Real data → maximize "goodness"  
- **Negative pass**: Fake data → minimize "goodness"

Each layer trains independently with its own objective. No gradient flow through the network.

📄 [Original Paper](https://www.cs.toronto.edu/~hinton/FFA13.pdf)

## Why This Fork?

We're exploring:
1. **CPU-only training** — Can we train without GPU?
2. **Efficiency improvements** — Early stopping, LR scheduling, better negatives
3. **Alternative architectures** — SSM layers instead of linear
4. **Text/sequence learning** — Beyond MNIST to language modeling
5. **Incremental learning** — Adding new data without full retraining

---

## Hardware Baseline

| Spec | Value |
|------|-------|
| **CPU** | Intel Xeon (Skylake) |
| **Cores** | 8 |
| **RAM** | 16 GB |
| **GPU** | None (CPU-only) |
| **PyTorch Threads** | 8 |

---

## Experiments Log

### Experiment 1: Quick CPU Baseline
**Date:** 2026-02-10  
**Goal:** Prove FF works on CPU

| Setting | Value |
|---------|-------|
| Architecture | `[784 → 500 → 500]` |
| Training samples | 5,000 (10% of MNIST) |
| Epochs per layer | 500 |
| Batch size | Full batch |
| Learning rate | 0.03 |
| Threshold | 2.0 |

**Results:**
| Metric | Value |
|--------|-------|
| Train error | 10.60% |
| Test error | **11.53%** |
| Train time | 66.9s |
| Total time | 77.1s |
| Peak memory | ~500 MB |
| CPU usage | ~670% (7 cores) |

**Conclusion:** FF works on CPU. High error due to reduced settings, not algorithm failure.

---

### Experiment 2: Full Paper Settings (In Progress)
**Date:** 2026-02-10  
**Goal:** Replicate Hinton's ~1.4% test error

| Setting | Value |
|---------|-------|
| Architecture | `[784 → 2000 → 2000 → 2000 → 2000]` |
| Training samples | 50,000 (full MNIST) |
| Epochs per layer | 500 |
| Batch size | Full batch |
| Learning rate | 0.03 |
| Threshold | 2.0 |

**Status:** 🔄 Running (Layer 1: ~60%)

**Expected:**
| Metric | Target |
|--------|--------|
| Test error | ~1.4% (paper) |
| Train time | ~3 hours |
| Peak memory | ~2 GB |

---

### Experiment 3: Early Stopping (Planned)
**Goal:** Reduce training time by stopping layers when converged

---

### Experiment 4: Text/Shakespeare (Planned)
**Goal:** Adapt FF for character-level language modeling

---

## Improvement Ideas

From Hinton's paper + our research:

1. **Unsquared goodness** — Use sum of activities, not squared
2. **Early stopping** — Move to next layer when loss plateaus
3. **LR scheduling** — Warmup + decay for faster convergence
4. **Harder negatives** — Use confused samples, not random
5. **SSM layers** — Replace linear with Mamba-style state spaces

## Files

| File | Description |
|------|-------------|
| `main.py` | Original implementation |
| `experiments/main_cpu.py` | Experiment 1: CPU baseline |
| `experiments/main_full.py` | Experiment 2: Full paper settings |
| `ANALYSIS.md` | Detailed research notes |

## Installation

```bash
pip install torch torchvision tqdm psutil
python experiments/main_cpu.py
```

## Citation

```bibtex
@article{hinton2022forward,
  title={The Forward-Forward Algorithm: Some Preliminary Investigations},
  author={Hinton, Geoffrey},
  journal={arXiv preprint arXiv:2212.13345},
  year={2022}
}
```

## License

MIT (same as original)

---

*Part of the [Local AI Manifesto](https://github.com/VictoriaDigital/engram) — training without the cloud.*
