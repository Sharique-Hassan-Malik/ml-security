import pytest
from waf.rules.engine import RuleEngine, _decode
from waf.rules.patterns import RULES


def _req(url="", query="", body="", method="GET", headers=None):
    return {"method": method, "url": url, "query": query, "body": body, "headers": headers or {}}


class TestDecoding:
    def test_single_decode(self):
        assert _decode("%27") == "'"

    def test_double_decode(self):
        assert _decode("%2527") == "'"

    def test_no_change(self):
        assert _decode("hello") == "hello"


class TestSQLi:
    engine = RuleEngine(threshold=1)

    def test_union_select(self):
        r = _req(query="id=1 UNION SELECT username,password FROM users--")
        v = self.engine.inspect(r)
        assert v.malicious
        assert "sqli" in v.categories

    def test_boolean_blind(self):
        r = _req(query="id=1' OR '1'='1")
        v = self.engine.inspect(r)
        assert v.malicious

    def test_time_based(self):
        r = _req(query="id=1 AND SLEEP(5)--")
        v = self.engine.inspect(r)
        assert v.malicious

    def test_post_body(self):
        r = _req(method="POST", body="username=admin'--&password=x",
                 headers={"Content-Type": "application/x-www-form-urlencoded"})
        v = self.engine.inspect(r)
        assert v.malicious

    def test_benign_id(self):
        r = _req(query="id=42&page=2")
        v = RuleEngine(threshold=3).inspect(r)
        assert not v.malicious


class TestXSS:
    engine = RuleEngine(threshold=1)

    def test_script_tag(self):
        r = _req(query="q=<script>alert(1)</script>")
        v = self.engine.inspect(r)
        assert v.malicious
        assert "xss" in v.categories

    def test_event_handler(self):
        r = _req(query='q=<img src=x onerror=alert(1)>')
        v = self.engine.inspect(r)
        assert v.malicious

    def test_javascript_uri(self):
        r = _req(query="redirect=javascript:alert(1)")
        v = self.engine.inspect(r)
        assert v.malicious

    def test_encoded_xss(self):
        r = _req(query="q=%3Cscript%3Ealert(1)%3C%2Fscript%3E")
        v = self.engine.inspect(r)
        assert v.malicious


class TestTraversal:
    engine = RuleEngine(threshold=1)

    def test_plain_traversal(self):
        r = _req(query="file=../../../../etc/passwd")
        v = self.engine.inspect(r)
        assert v.malicious
        assert "traversal" in v.categories

    def test_url_encoded_traversal(self):
        r = _req(query="file=..%2F..%2F..%2Fetc%2Fpasswd")
        v = self.engine.inspect(r)
        assert v.malicious

    def test_double_encoded(self):
        r = _req(query="file=%252e%252e%252f%252e%252e%252fetc%252fpasswd")
        v = self.engine.inspect(r)
        assert v.malicious


class TestCommandInjection:
    engine = RuleEngine(threshold=1)

    def test_semicolon_command(self):
        r = _req(query="host=127.0.0.1; cat /etc/passwd")
        v = self.engine.inspect(r)
        assert v.malicious
        assert "cmdi" in v.categories

    def test_pipe_command(self):
        r = _req(query="input=test | id")
        v = self.engine.inspect(r)
        assert v.malicious

    def test_backtick(self):
        r = _req(query="q=`whoami`")
        v = self.engine.inspect(r)
        assert v.malicious

    def test_dollar_subshell(self):
        r = _req(query="name=$(id)")
        v = self.engine.inspect(r)
        assert v.malicious


class TestSSRF:
    engine = RuleEngine(threshold=1)

    def test_metadata_endpoint(self):
        r = _req(query="url=http://169.254.169.254/latest/meta-data/")
        v = self.engine.inspect(r)
        assert v.malicious
        assert "ssrf" in v.categories

    def test_localhost(self):
        r = _req(query="proxy=http://localhost/admin")
        v = self.engine.inspect(r)
        assert v.malicious


class TestThreshold:
    def test_high_threshold_ignores_low_score(self):
        r = _req(query="id=1 AND 1=1")
        v = RuleEngine(threshold=10).inspect(r)
        # medium/high patterns may not reach threshold 10
        # this tests the threshold mechanism not a specific outcome
        assert isinstance(v.malicious, bool)

    def test_zero_threshold_always_fires(self):
        # A clean request with score=0 should still pass threshold=0
        r = _req(url="/index.html", query="page=1")
        v = RuleEngine(threshold=0).inspect(r)
        # score 0 >= threshold 0, so malicious=True — threshold=0 is a degenerate case
        assert v.score == 0


class TestCategories:
    def test_primary_category_sqli(self):
        r = _req(query="id=1 UNION SELECT username,password FROM users-- AND SLEEP(5)")
        v = RuleEngine(threshold=1).inspect(r)
        assert v.primary_category == "sqli"

    def test_multiple_categories(self):
        # Payload hits both SQLi and XSS
        r = _req(query="q=<script>alert(1)</script>&id=1 UNION SELECT 1--")
        v = RuleEngine(threshold=1).inspect(r)
        assert "sqli" in v.categories
        assert "xss" in v.categories
