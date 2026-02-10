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

## Experiments

| File | Description | Status |
|------|-------------|--------|
| `experiments/main_cpu.py` | CPU-only baseline | ✅ Works |
| `experiments/main_full.py` | Full paper settings (4 layers, 2000 neurons) | 🔄 Running |
| `experiments/early_stopping.py` | Stop layers when converged | TODO |
| `experiments/lr_schedule.py` | Warmup + cosine decay | TODO |
| `experiments/text_ff.py` | Character-level language model | TODO |

## Results

### Baseline (CPU, reduced settings)
```
Architecture: [784 → 500 → 500]
Samples: 5,000
Epochs: 500/layer
Test error: 11.53%
Time: 77 seconds
```

### Full Settings (CPU, in progress)
```
Architecture: [784 → 2000 → 2000 → 2000 → 2000]
Samples: 50,000
Epochs: 500/layer
Target: ~1.4% test error
Time: ~3 hours (estimated)
```

## Improvement Ideas

From Hinton's paper + our research:

1. **Unsquared goodness** — Use sum of activities, not squared
2. **Early stopping** — Move to next layer when loss plateaus
3. **LR scheduling** — Warmup + decay for faster convergence
4. **Harder negatives** — Use confused samples, not random
5. **SSM layers** — Replace linear with Mamba-style state spaces

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
