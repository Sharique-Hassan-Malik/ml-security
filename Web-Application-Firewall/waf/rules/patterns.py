import re

# Each entry: (category, severity, compiled_pattern, description)
# Patterns are applied to the full URL-decoded request string.

_RAW: list[tuple[str, str, str, str]] = [
    # SQL Injection
    ("sqli", "critical",
     r"(?i)\b(union\s+all\s+select|union\s+select)\b",
     "UNION SELECT statement"),
    ("sqli", "critical",
     r"(?i)\b(select\s+.+\s+from\s+|insert\s+into\s+|update\s+.+\s+set\s+|delete\s+from\s+|drop\s+table\s+|create\s+table\s+|alter\s+table\s+)",
     "SQL DML/DDL keyword sequence"),
    ("sqli", "high",
     r"(?i)('\s*(or|and)\s*'?\d|\d\s*(or|and)\s*\d\s*=\s*\d)",
     "Boolean-based blind SQLi"),
    ("sqli", "high",
     r"(?i)(--\s*$|;\s*--|#\s*$|\*\/|\\/\*)",
     "SQL comment terminator"),
    ("sqli", "high",
     r"(?i)\b(sleep\s*\(|benchmark\s*\(|waitfor\s+delay\s+)",
     "Time-based blind SQLi"),
    ("sqli", "medium",
     r"(?i)(0x[0-9a-f]{2,}|char\s*\(\s*\d+\s*\))",
     "Hex/char encoding in SQL context"),
    ("sqli", "medium",
     r"(?i)\b(information_schema|sysobjects|syscolumns|pg_sleep|xp_cmdshell)\b",
     "SQL system object reference"),

    # XSS
    ("xss", "critical",
     r"(?i)<\s*script[^>]*>",
     "Script tag injection"),
    ("xss", "critical",
     r"(?i)\bon\w+\s*=\s*['\"]?\s*(javascript|script|vbscript|\w+\s*\()",
     "Inline event handler with script"),
    ("xss", "high",
     r"(?i)(javascript|vbscript|data\s*:)\s*:",
     "javascript: or data: URI scheme"),
    ("xss", "high",
     r"(?i)<\s*(iframe|object|embed|applet|link|meta|base)[^>]*>",
     "Dangerous HTML tag injection"),
    ("xss", "high",
     r"(?i)expression\s*\(",
     "CSS expression injection"),
    ("xss", "medium",
     r"(?i)(\\u003c|\\u003e|\\x3c|\\x3e|%3cscript|%3c%2fscript)",
     "Unicode/hex-encoded angle brackets"),
    ("xss", "medium",
     r"(?i)(document\.(cookie|write|location)|window\.(location|open)|eval\s*\()",
     "DOM manipulation sink"),

    # Path Traversal
    ("traversal", "critical",
     r"(\.\.[\\/]){2,}",
     "Directory traversal sequence"),
    ("traversal", "critical",
     r"(%2e%2e[\\/]|%2e%2e%2f|%252e%252e)",
     "URL-encoded traversal sequence"),
    ("traversal", "high",
     r"(?i)(\/etc\/passwd|\/etc\/shadow|\/proc\/self|\/windows\/win\.ini|boot\.ini)",
     "Sensitive file path reference"),
    ("traversal", "high",
     r"(?i)(\.\./|\.\.\\){1,}(etc|windows|proc|sys|var)",
     "Traversal to system directory"),

    # Command Injection
    ("cmdi", "critical",
     r"(?i)(;\s*(ls|cat|id|whoami|uname|pwd|curl|wget|nc|bash|sh|python|perl|php)\b)",
     "Shell command after semicolon"),
    ("cmdi", "critical",
     r"(`[^`]*`|\$\([^)]+\))",
     "Command substitution (backtick or $())"),
    ("cmdi", "critical",
     r"(?i)(\|\s*(ls|cat|id|whoami|bash|sh|nc|curl|wget))",
     "Piped shell command"),
    ("cmdi", "high",
     r"(?i)(&&\s*(ls|cat|id|whoami|bash|sh)|;\s*echo\s+)",
     "Chained shell command"),
    ("cmdi", "high",
     r"(?i)(%0a|%0d|%3b)\s*(ls|cat|id|bash|sh|curl|wget)",
     "Newline/semicolon-injected shell command"),

    # SSRF
    ("ssrf", "critical",
     r"(?i)(https?|ftp|file|gopher|ldap|dict|tftp):\/\/(127\.0\.0\.1|localhost|0\.0\.0\.0|169\.254\.169\.254)",
     "SSRF to loopback or metadata endpoint"),
    ("ssrf", "high",
     r"(?i)169\.254\.169\.254",
     "AWS/cloud metadata endpoint"),
    ("ssrf", "high",
     r"(?i)(file:\/\/\/|file:\/\/localhost)",
     "file:// URI scheme"),

    # XXE
    ("xxe", "critical",
     r"(?i)(<!ENTITY\s+\w+\s+SYSTEM\s+['\"]|<!DOCTYPE[^>]*\[)",
     "XXE entity declaration"),
    ("xxe", "high",
     r"(?i)SYSTEM\s+['\"]file:\/\/",
     "XXE file:// system entity"),
]

# Compiled rule objects
class Rule:
    __slots__ = ("category", "severity", "pattern", "description")

    def __init__(self, category: str, severity: str, pattern: re.Pattern, description: str):
        self.category    = category
        self.severity    = severity
        self.pattern     = pattern
        self.description = description

    def match(self, text: str) -> re.Match | None:
        return self.pattern.search(text)


RULES: list[Rule] = [
    Rule(cat, sev, re.compile(pat), desc)
    for cat, sev, pat, desc in _RAW
]

SEVERITY_SCORE = {"critical": 4, "high": 3, "medium": 2, "low": 1}
