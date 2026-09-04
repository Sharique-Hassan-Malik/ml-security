# Architecture

Six modules joined by one contract. This document is about the contract and the
decisions behind it; each module's own analysis is documented in
[`docs/`](docs).

## The shape

```
                       aisec/cli.py
                            │
                       aisec/runner.py ──── fans a target across every
                            │               module that claims it
              ┌─────────────┼─────────────┐
              │             │             │
         scanners       guards        probes
              │             │             │
        ┌─────┴─────┐  ┌────┴────┐  ┌─────┴─────┐
    pickle-     weight-  prompt-  adversarial  model-   gradient-
    scanner   poisoning injection  detection  extraction leakage
              └───────────────┬────────────────┘
                              │
                    aisec/core/finding.py
                    one Severity, one Finding, one Report
```

Modules never talk to the CLI and never render anything. They return a
`ModuleResult`. Everything downstream — terminal output, JSON, HTML, exit
codes, the aggregate verdict — is computed from that one type.

## Why three kinds and not one

A `Kind` is not a label, it is a different call signature:

- **Scanner** — `scan(path) -> ModuleResult`. Offline, over an artifact, with
  no model execution.
- **Guard** — `inspect(payload) -> ModuleResult`. In a live request path, so it
  must be cheap and must never raise into the caller.
- **Probe** — `run(**options) -> ModuleResult`. An attack against a model you
  own, expensive by nature, and never run implicitly.

Forcing one `run()` across all three would mean a scanner accepting a path it
ignores, or a guard returning a report nobody is waiting for. The runner knows
which shape to call, so modules do not have to pretend to be interchangeable.

Guards additionally declare what they can be handed. A text firewall and a
tensor detector are both guards, and running one on the other's input would
produce a confident answer about nothing — so `ModuleSpec.payloads` is checked
first and the mismatch is reported as a skip:

```
  adversarial-detection  (guard)
    skipped — takes tensor, got text
```

## Severity is one ladder

`SAFE < INFO < LOW < MEDIUM < HIGH < CRITICAL`, ordered and comparable, so
`finding.severity >= Severity.HIGH` means the same thing in a bytecode scanner
and in a gradient-inversion probe. The alternative is what these tools reach
for individually — strings `"low"/"medium"/"high"` in one, a float in [0,1] in
another, an enum in a third — and findings that cannot be compared across tools
make a single verdict impossible.

The report's verdict is the **worst** severity present, never an average.
Averaging is how one CRITICAL finding gets diluted by nine clean checks into
something that looks survivable.

`metrics` is the escape hatch that keeps the shared schema from flattening
everything into findings. An extraction probe's headline is "0.94 agreement
after 3,000 queries" and a leakage probe's is "PSNR 31.2 dB" — neither is a
finding, both belong in the report.

## Nothing is imported until it is needed

`aisec/core/registry.py` holds a static `MANIFEST` of `ModuleSpec` values. It
is data, not the product of importing anything. `aisec list` and
`aisec scan model.pkl` therefore run with no torch installed, and a module
whose dependencies are missing is listed as `needs torch` rather than crashing
the process on import.

Loading is by file path, not by dotted module name: each module folder is its
own source root (`modules/pickle-scanner` contains the `pickle_scanner`
package), so the folder goes on `sys.path` and its `integration.py` is imported
under a unique key. That is what lets every module keep an `integration.py`
without colliding, and it is the same import path a module gets when run
standalone from its own directory — so "works standalone" and "works in the
platform" cannot drift apart.

## Standalone and integrated, without duplication

The requirement was that each tool remain individually usable. The trap is
solving that by copying the shared vocabulary into each module, which
guarantees the copies diverge.

Instead `aisec/core` is **stdlib-only**. A module importing the shared
`Severity` and `Finding` pays nothing for it, so `cd modules/pickle-scanner &&
python scan.py x.pkl` works with no install, no torch, and no platform code
beyond the schema and the renderer. Each standalone CLI adds its module folder
and the repo root to `sys.path` in two lines and is otherwise ordinary.

The orchestration lives in exactly one place per module: `integration.py` holds
the run logic, and the standalone CLI is a thin argument parser over it. The
probe you get from `aisec probe model-extraction` and the one you get from
`python experiment.py` are the same code path, so the benchmark cannot drift
from the thing being benchmarked.

## Modules that use each other

`adversarial-detection` needs the model it is protecting, and loading a
checkpoint means unpickling it — precisely what `pickle-scanner` exists to warn
about. So it screens the file with the sibling scanner first and refuses
anything rated HIGH or worse, rather than being the tool in the security suite
that executes the payload.

The lookup goes through `registry.module_path()` rather than a relative path,
and falls back to a plain load if the sibling is absent: a guard should not
become unusable because another module was removed.

## Rendering

One renderer, in `aisec/core/render.py`, replacing three near-identical HTML
generators. Modules emit findings and, when they have a curve worth showing, a
declarative chart spec in `metrics["charts"]` — they do not draw.

Colour rules that the output has to satisfy:

- Severity ships as **glyph + written name**, with colour only reinforcing.
  The status hexes are fixed by the palette and two of them sit below 3:1 on a
  light surface by design; the icon-and-label pairing is the required
  mitigation, and it is also what makes the terminal output readable once a CI
  log has stripped the ANSI codes.
- Chart series take the eight categorical slots in fixed order, never cycled.
  Line charts compare neighbouring series, so all eight are in play; an
  all-pairs form would be capped at three. Up to four series are direct
  labelled, beyond that a legend row carries the names — identity is never left
  to colour alone.
- Light and dark are both defined explicitly, under bare `:root`, under
  `prefers-color-scheme`, and under `[data-theme]`, so a theme toggle wins in
  both directions.

## Known trade-offs

- **The heuristic layer of the prompt firewall is the weakest of its three
  controls** and gets the most attention. Channel separation and tool
  allowlisting carry the load, because they do not depend on recognising
  anything. The module's own docs are explicit about this rather than quoting a
  detection rate as if it were a guarantee.
- **The bundled model architectures are small on purpose.** LeNet-5 and a
  three-block CNN converge in seconds, which is what makes the probes runnable
  as tests. Numbers from them describe the attack's behaviour, not your
  production model's exposure.
- **`weights_only=True` is used wherever a tensor is loaded**, but loading a
  full `nn.Module` cannot be done that way — hence the pickle screen described
  above rather than a claim of safety.
