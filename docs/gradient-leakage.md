# Architecture

---

## Module Map

```
experiment.py          (CLI entry point — end-to-end demo)
│
├── attack/
│   ├── inversion.py   DLG / iDLG optimisation attack
│   │     AttackConfig, AttackResult
│   │     GradientInversionAttack.attack()
│   │     compute_observed_gradients()
│   └── rgap.py        R-GAP closed-form attack (MLP only)
│         RGAPAttack.attack()
│
├── defense/
│   └── defenses.py
│         DifferentialPrivacyDefense   (clip + Gaussian noise)
│         GradientCompressionDefense   (top-k sparsification)
│         GradientNoiseDefense         (additive Gaussian/Laplace)
│         gradient_cosine_similarity()
│         gradient_snr()
│
├── metrics/
│   └── quality.py
│         psnr(), ssim(), mse()
│         reconstruction_quality()
│
└── visualize/
    └── report.py      Self-contained HTML report generator
          generate_html_report()
```

---

## Attack Data Flow

```
Real training data (x, y)
        │
        ▼
compute_observed_gradients()
        │  torch.autograd.grad(ℒ(F(x), y), model.parameters())
        ▼
List[Tensor]  ← this is what the server sees
        │
        ▼
GradientInversionAttack.attack()
        │
        ├─ iDLG: recover y from argmin(col_sum(∂ℒ/∂W_last))
        │
        ├─ Initialise x̃ ~ N(0,1),  ỹ = recovered or random
        │
        └─ L-BFGS loop:
               dummy_grads = ∇ℒ(F(x̃), ỹ)
               loss = ‖dummy_grads − observed_grads‖² + TV(x̃)
               x̃ ← x̃ − α ∇_{x̃} loss
        │
        ▼
AttackResult (dummy_data, dummy_labels, losses)
```

---

## Defense Data Flow

```
observed_grads
        │
        ▼
defense.apply(grads) → (defended_grads, metadata)
        │
        ├── DP:          clip(g, C)  +  N(0, σ²C²)
        ├── Compression: zero all |g_i| < percentile(|g|, sparsity)
        └── Noise:       g  +  N(0, σ²)  or  Laplace(0, σ)
        │
        ▼
attacker.attack(defended_grads, ...)
        │
        ▼
reconstruction quality metrics (PSNR, SSIM, MSE)
```

---

## Privacy Guarantee (DP-SGD)

With `max_grad_norm = C` and `noise_multiplier = σ`:

- Per-step RDP (order α=2): `ε_step = α / (2σ²)`
- After T steps: `ε_total ≈ T × ε_step`
- Convert to (ε, δ)-DP: `ε = ε_total + log(1/δ) / (α - 1)`

For production use, apply the moments accountant or PRV accountant
(Gopi et al. 2021) for tighter bounds.
