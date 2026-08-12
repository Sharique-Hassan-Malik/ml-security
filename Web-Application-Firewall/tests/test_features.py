import math
import pytest
from waf.ml.features import extract, FEATURE_NAMES


def _req(url="", query="", body="", method="GET", headers=None):
    return {"method": method, "url": url, "query": query, "body": body, "headers": headers or {}}


class TestFeatureVectorShape:
    def test_length(self):
        r = _req(url="/index.html", query="id=1")
        assert len(extract(r)) == 38

    def test_length_matches_names(self):
        assert len(FEATURE_NAMES) == 38

    def test_all_floats(self):
        r = _req(url="/search?q=test")
        vec = extract(r)
        assert all(isinstance(v, float) for v in vec)

    def test_empty_request(self):
        vec = extract({})
        assert len(vec) == 38
        assert all(math.isfinite(v) for v in vec)


class TestSpecificFeatures:
    def test_url_length(self):
        url = "/product/detail"
        vec = extract(_req(url=url))
        assert vec[FEATURE_NAMES.index("url_len")] == float(len(url))

    def test_param_count(self):
        r = _req(query="a=1&b=2&c=3")
        vec = extract(r)
        assert vec[FEATURE_NAMES.index("param_count")] == 3.0

    def test_sql_kw_count_positive(self):
        r = _req(query="id=1 UNION SELECT password FROM users")
        vec = extract(r)
        assert vec[FEATURE_NAMES.index("sql_kw_count")] >= 3

    def test_xss_kw_count_positive(self):
        r = _req(query="q=<script>alert(1)</script>")
        vec = extract(r)
        assert vec[FEATURE_NAMES.index("xss_kw_count")] >= 1

    def test_traversal_kw_positive(self):
        r = _req(query="file=../../etc/passwd")
        vec = extract(r)
        assert vec[FEATURE_NAMES.index("traversal_kw_count")] >= 1

    def test_single_quote_count(self):
        r = _req(query="id=1' OR '1'='1")
        vec = extract(r)
        assert vec[FEATURE_NAMES.index("single_quote_count")] == 4.0

    def test_method_score_post(self):
        r = _req(method="POST", url="/submit")
        vec = extract(r)
        assert vec[FEATURE_NAMES.index("method_score")] == 1.0

    def test_method_score_get(self):
        r = _req(method="GET", url="/home")
        vec = extract(r)
        assert vec[FEATURE_NAMES.index("method_score")] == 0.0

    def test_entropy_increases_with_complexity(self):
        simple  = _req(query="q=aaaa")
        complex = _req(query="q=aB3!@#d")
        e_s = extract(simple)[FEATURE_NAMES.index("query_entropy")]
        e_c = extract(complex)[FEATURE_NAMES.index("query_entropy")]
        assert e_c > e_s

    def test_pct_encoded_count(self):
        r = _req(url="/file?name=%2e%2e%2f%2e%2e%2fetc%2fpasswd")
        vec = extract(r)
        assert vec[FEATURE_NAMES.index("url_pct_encoded")] >= 6

    def test_special_ratio_attack_higher_than_benign(self):
        # Use body field to avoid the & and = chars in benign query strings inflating the ratio
        benign = _req(body="username=alice&password=secret123")
        attack = _req(body="q=<script>alert(document.cookie)</script>|`id`;echo$IFS'pwned'")
        b_vec  = extract(benign)
        a_vec  = extract(attack)
        idx    = FEATURE_NAMES.index("body_special_ratio")
        assert a_vec[idx] > b_vec[idx]
