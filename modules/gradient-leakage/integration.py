"""Joins gradient-leakage to the suite as a probe.

Severity is keyed to reconstruction PSNR against the original image, because
that is what an operator can act on: 25 dB is a recognisable photograph, 10 dB
is noise. The defended runs are then judged against the same bar — a defence
that leaves PSNR where it was is reported as a finding in its own right, since
"we enabled DP" is otherwise indistinguishable from "we are protected".

Every defence's cost is reported beside its benefit. A defence that stops
inversion by destroying the gradient has not solved the problem.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
for _path in (_HERE, _HERE.parents[1]):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from aisec.core.finding import Finding, ModuleResult, Severity  # noqa: E402
from aisec.core.module import Probe  # noqa: E402
from aisec.core.registry import spec  # noqa: E402
from aisec.core.render import png_data_uri  # noqa: E402
from gradleak import DEFENCES, imaging, run_leakage  # noqa: E402

# dB thresholds. 20 dB is roughly where the subject of a 32x32 image becomes
# identifiable; below 10 dB there is nothing left to recognise.
_PSNR_LADDER = ((25.0, Severity.CRITICAL), (20.0, Severity.HIGH),
                (15.0, Severity.MEDIUM), (10.0, Severity.LOW))


def _severity(psnr: float) -> Severity:
    for threshold, severity in _PSNR_LADDER:
        if psnr >= threshold:
            return severity
    return Severity.INFO


class GradientLeakageModule(Probe):
    def run(self, **options: Any) -> ModuleResult:
        result = self.result(str(options.get("label", "federated client")))

        defences = tuple(options.get("defences") or DEFENCES)
        iterations = int(options.get("iterations", 300))

        real, runs = run_leakage(
            iterations=iterations,
            algorithm=str(options.get("algorithm", "idlg")),
            seed=int(options.get("seed", 42)),
            defences=defences,
        )

        baseline = next((r for r in runs if r.name == "no defence"), None)

        for run in runs:
            undefended = run is baseline
            severity = _severity(run.psnr)
            summary = f"reconstruction PSNR {run.psnr:.1f} dB, MSE {run.mse:.4f}"

            if undefended:
                title = "gradient_inversion"
                detail = (
                    "Shared gradients alone were enough to reconstruct the client's "
                    "input. No model weights and no data were transmitted."
                )
            else:
                title = f"defence_{run.name.replace(' ', '_')}"
                if baseline is not None and run.psnr >= baseline.psnr - 2.0:
                    # Within noise of the undefended run: it did not help.
                    detail = (
                        f"Reconstruction is no worse than undefended "
                        f"({baseline.psnr:.1f} dB) — this setting does not stop the "
                        f"attack. Gradient similarity to the true gradient is "
                        f"{run.cosine_sim:.3f}."
                    )
                else:
                    severity = min(severity, Severity.MEDIUM, key=lambda s: s.rank)
                    detail = (
                        f"Reconstruction degraded from {baseline.psnr:.1f} dB to "
                        f"{run.psnr:.1f} dB. Cost: gradient cosine similarity "
                        f"{run.cosine_sim:.3f} (utility loss {run.utility_cost:.3f})."
                        if baseline is not None else ""
                    )

            result.add(
                Finding(
                    title=title,
                    severity=severity,
                    summary=summary,
                    detail=detail,
                    location=run.name,
                    score=round(min(run.psnr / 30.0, 1.0), 4),
                    metadata={
                        "psnr_db": round(run.psnr, 3),
                        "ssim": None if run.ssim is None else round(run.ssim, 4),
                        "mse": round(run.mse, 6),
                        "gradient_cosine": round(run.cosine_sim, 4),
                        "gradient_snr_db": (
                            None if run.snr_db == float("inf") else round(run.snr_db, 2)
                        ),
                    },
                )
            )

        result.metrics["iterations"] = iterations
        result.metrics["defences"] = ", ".join(r.name for r in runs)
        if baseline is not None:
            result.metrics["undefended_psnr_db"] = round(baseline.psnr, 2)
        result.metrics["charts"] = [
            {
                "title": "Inversion loss per iteration — lower means a better reconstruction",
                "series": {
                    run.name: list(enumerate(run.losses))
                    for run in runs if run.losses
                },
                "x_label": "Iteration",
                "y_label": "Gradient matching loss",
            }
        ]

        if imaging.available():
            images = [{"uri": png_data_uri(imaging.tensor_to_png(real)),
                       "caption": "original (private) input"}]
            images += [
                {"uri": png_data_uri(imaging.tensor_to_png(run.reconstruction[0])),
                 "caption": f"{run.name} — {run.psnr:.1f} dB"}
                for run in runs
            ]
            result.metrics["images"] = images

        return result


MODULE = GradientLeakageModule(spec("gradient-leakage"))
