# Architecture

## Three controls, in order of how much they carry

```
user message ──▶ Firewall.check(source="user")          heuristic
                      │
retrieved doc ──▶ Firewall.process_document()           STRUCTURAL
                      │  sanitise → score → fence
                      ▼
                  prompt assembled: instructions | <<fence>> data <<fence>>
                      │
                      ▼
                  model
                      │
   tool call ──▶ ToolGuard.check()                      STRUCTURAL
                      │  allowlist → schema → value policy
                      ▼
   response  ──▶ OutputFilter.inspect()                 STRUCTURAL
                         auto-fetch → links → secrets
```

Only the first box is detection. It is placed first in the request path and last
in importance, and the code says so in its own module docstring — because the
failure mode of this whole category of tool is a team that ships a detector and
believes the problem is handled.

## Why detection cannot be the answer

An HTTP firewall works because HTTP has a grammar: a header is not a body, and
SQL in a `WHERE` clause has a shape. An LLM prompt has no grammar. "Ignore all
previous instructions" is an attack in one context and a sentence in a security
paper in another, and the bytes are identical.

The corpus makes this concrete. Three benign samples are users *reporting* an
attack they received. The detector flags all three, and no threshold fixes it,
because the report quotes the attack. `discussion_discount()` lowers the score
for text that reads like analysis — hedged, cited, hypothetical — which helps
with papers and not at all with someone pasting the payload and asking "is this
bad?".

The honest conclusion is in the base-rate table: at realistic prevalence this
detector yields ~57 alerts per real attack. That is a logging signal, not a
blocking control, and the number is the reason.

## Structural control 1: channel separation

```python
def wrap_untrusted(self, content, label="document"):
    cleaned = content.replace(self.fence, "")           # ← the load-bearing line
    cleaned = re.sub(r"<\|(im_start|im_end|...)\|>", "", cleaned)
    return f"{self.fence}\n...UNTRUSTED DATA...\n{cleaned}\n{self.fence}"
```

Stripping the fence token from the content is the whole mechanism. Without it,
content containing the fence closes it, and everything after is read as
instruction — delimiter injection in one line. A test asserts the wrapped output
contains exactly two fences.

This holds against attacks nobody has catalogued, because it does not inspect
the content's meaning at all.

## Structural control 2: tool allowlisting

Detection asks "is this text an attack?", which has no reliable answer.
Allowlisting asks "is this call permitted?", which has an exact one.

| layer | question | example denial |
|---|---|---|
| allowlist | is the tool callable? | `exec` is not on the list |
| schema | do arguments type-check? | `query` must be `str` |
| value policy | are the *values* in bounds? | `/.ssh/id_rsa` escapes the sandbox |

The third layer is where the real incidents live. Path normalisation happens
*before* the prefix check, because the entire point of `../..` is that the raw
string passes a check the resolved path fails. Domain matching compares host
labels rather than string suffixes, because `notexample.com` ends with
`example.com`.

Default-deny is not a style choice: an allowlist that falls open on an unknown
tool is a denylist.

## Structural control 3: output filtering

Ordered by how little user action is required:

1. **auto-fetched constructs** — markdown images, `<img>`, `<script>`,
   `<iframe>`, CSS `url()`. The client fetches these on render. The request
   itself is the exfiltration; nothing needs to be "said".
2. **suspicious links** — an out-of-domain URL with a long opaque query string
   is the shape of encoded data whatever it claims to be.
3. **secrets** — format-based matching. Cheap, and unreliable by construction:
   it finds keys shaped like keys and misses everything else.

Redaction walks findings right-to-left so earlier spans stay valid after
replacement.

## Detector design notes

**Max, not sum.** `Firewall.score` takes the maximum weighted signal. Summing
lets several weak independent signals accumulate, which makes any sufficiently
long text look suspicious — a failure mode that grows with input size.

**Decode before matching.** `detect_encoded_payload` base64- and hex-decodes
candidates and runs the instruction detectors on the *result*. A detector that
only reads the surface form is defeated by one `base64.b64encode`.

**Obfuscation is the highest-precision detector.** Zero-width characters and
bidirectional overrides appear in essentially no legitimate prose, and bidi has
exactly one use in a prompt: making rendered text differ from parsed text. This
is also the only detector whose findings are *sanitised* rather than scored,
since stripping them is lossless.

**Documents are held to a stricter threshold than users.** A user instructing
the assistant is normal; a web page instructing the assistant never is. Same
detectors, thresholds multiplied by 0.65.

## The corpus

Attacks are labelled by family so recall can be broken down — a detector at 80%
overall that catches 0% of encoded payloads is a different and much worse
product than one that misses evenly.

The benign half is the part that makes the numbers mean anything, and it is
deliberately adversarial: security writing, incident reports, users quoting
attacks, and technical content with suspicious shapes (JWTs, base64 config,
`{"role": "system"}` in a code sample). Against attacks paired with "what is the
capital of France?", any detector scores near-perfectly and has been measured
for reading comprehension rather than discrimination.

## Layout

| path | role |
|---|---|
| `pifw/detectors.py` | six scoring heuristics, and the discussion discount |
| `pifw/firewall.py` | scoring, channel separation, sanitisation |
| `pifw/tools.py` | three-layer allowlisting |
| `pifw/output.py` | exfiltration filtering and redaction |
| `pifw/corpus.py` | attacks by family, benign by why-it-is-hard |
| `pifw/evaluate.py` | confusion matrices and the base-rate projection |
| `bench/evaluate.py` | the report |

## What is out of scope

- **Model-based detection.** Likely better recall, identical base-rate problem.
- **Multi-turn attacks.** A payload split across turns defeats every per-message
  detector here.
- **Multi-modal injection.** Image alt-text and OCR'd instructions are real and
  not addressed.
- **Rate limiting and quarantine.** The natural consumers of a `FLAG` decision,
  and the reason `FLAG` exists as distinct from `BLOCK`.
