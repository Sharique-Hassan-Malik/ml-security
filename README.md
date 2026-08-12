# ML Security

Attacks and defenses on machine-learning systems: adversarial examples, model extraction, gradient leakage, weight poisoning, and firewalls for the LLM and web layers. Each detector is evaluated with precision/recall against an attack corpus, not asserted.

A collection of 6 self-contained projects. Each lives in its own subdirectory with its own `README.md` and `LICENSE` (most also include an `ARCHITECTURE.md` and a test suite), and can be built and run independently.

## Projects

| project | what it is |
|---|---|
| [`Adversarial-Detection`](./Adversarial-Detection) | Generates adversarial examples with FGSM, PGD, C&W and AutoAttack, then detects them using three independent methods fused through a unified scorin… |
| [`Gradient-Leakage-Analyzer`](./Gradient-Leakage-Analyzer) | Implements federated learning gradient inversion attacks (DLG and iDLG) and three defenses, with quantitative reconstruction quality metrics and an… |
| [`Model-Extraction`](./Model-Extraction) | Given only black-box API access to a victim model, reconstruct a functionally equivalent substitute model. |
| [`Prompt-Injection-Firewall`](./Prompt-Injection-Firewall) | An LLM-layer WAF: injection and jailbreak detection, tool-call allowlisting, output exfiltration filtering, and an attack corpus with precision/rec… |
| [`Web-Application-Firewall`](./Web-Application-Firewall) | A WAF that classifies HTTP requests as benign or malicious using two parallel approaches — a deterministic rule engine and a trained ML classifier… |
| [`Weight-Poisoning-Detector`](./Weight-Poisoning-Detector) | Static analysis tool that scans PyTorch .pt / .pth model files for backdoor attack signatures — no inference, no training data required. |

## Repository layout

Each subdirectory is a standalone project; there is no shared build. Enter one and follow its README:

```bash
cd Adversarial-Detection
cat README.md
```

## License

MIT — see the `LICENSE` file in each project.
