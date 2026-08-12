"""prompt-injection-firewall -- an LLM-layer WAF, and an honest account of its limits.

    from pifw import Firewall, ToolGuard, OutputFilter, default_policies

    firewall = Firewall()
    decision = firewall.check(user_message)                 # heuristic
    prompt   = firewall.process_document(retrieved).text    # structural
    verdict  = ToolGuard(default_policies()).check(call)    # structural
    outbound = OutputFilter(("example.com",)).inspect(reply)

Detection is the weakest of the three controls and gets the most attention;
`ARCHITECTURE.md` explains why the other two carry the load.
"""

from .corpus import ATTACKS, BENIGN, Sample, all_samples, counts
from .detectors import Signal, scan
from .evaluate import Confusion, Evaluation, OperatingPoint, evaluate, operating_point, sweep
from .firewall import Action, Decision, Firewall
from .output import OutputFilter, OutputFinding, OutputVerdict
from .tools import ToolCall, ToolGuard, ToolPolicy, ToolVerdict, default_policies

__version__ = "1.0.0"
__all__ = [
    "ATTACKS", "BENIGN", "Sample", "all_samples", "counts",
    "Signal", "scan",
    "Confusion", "Evaluation", "OperatingPoint", "evaluate", "operating_point", "sweep",
    "Action", "Decision", "Firewall",
    "OutputFilter", "OutputFinding", "OutputVerdict",
    "ToolCall", "ToolGuard", "ToolPolicy", "ToolVerdict", "default_policies",
]
