import unittest

from kreguard.permissions import EgressPolicy, ToolPolicy, ToolRule
from kreguard.verdict import Verdict


class ToolPolicyTests(unittest.TestCase):
    def test_empty_policy_denies_everything(self):
        self.assertEqual(ToolPolicy().authorize("search").verdict, Verdict.BLOCK)

    def test_allowlisted_tool_allowed(self):
        p = ToolPolicy([ToolRule("search")])
        self.assertEqual(p.authorize("search", {"q": "x"}).verdict, Verdict.ALLOW)
        self.assertEqual(p.authorize("delete_db").verdict, Verdict.BLOCK)

    def test_deny_wins_over_allow(self):
        p = ToolPolicy([ToolRule("shell")]).deny("shell")
        self.assertEqual(p.authorize("shell").verdict, Verdict.BLOCK)

    def test_validator_rejects_with_reason(self):
        p = ToolPolicy([ToolRule("send_email", validate=lambda a: "external recipient" if not a.get("to", "").endswith("@acme.com") else True)])
        d = p.authorize("send_email", {"to": "x@evil.com"})
        self.assertEqual(d.verdict, Verdict.BLOCK)
        self.assertEqual(d.findings[0].detail, "external recipient")
        self.assertEqual(p.authorize("send_email", {"to": "x@acme.com"}).verdict, Verdict.ALLOW)

    def test_validator_exception_is_block(self):
        def boom(a):
            raise KeyError("missing")

        p = ToolPolicy([ToolRule("t", validate=boom)])
        self.assertEqual(p.authorize("t").verdict, Verdict.BLOCK)

    def test_validator_weird_return_is_block(self):
        p = ToolPolicy([ToolRule("t", validate=lambda a: 1)])
        self.assertEqual(p.authorize("t").verdict, Verdict.BLOCK)

    def test_confirmation_flags(self):
        p = ToolPolicy([ToolRule("refund", confirm=True)])
        self.assertEqual(p.authorize("refund").verdict, Verdict.FLAG)

    def test_budget(self):
        p = ToolPolicy([ToolRule("refund", max_calls=2)])
        self.assertEqual(p.authorize("refund").verdict, Verdict.ALLOW)
        self.assertEqual(p.authorize("refund").verdict, Verdict.ALLOW)
        self.assertEqual(p.authorize("refund").verdict, Verdict.BLOCK)
        p.reset_budgets()
        self.assertEqual(p.authorize("refund").verdict, Verdict.ALLOW)

    def test_bad_names(self):
        p = ToolPolicy([ToolRule("t")])
        self.assertEqual(p.authorize("").verdict, Verdict.BLOCK)
        self.assertEqual(p.authorize(None).verdict, Verdict.BLOCK)  # type: ignore[arg-type]


class EgressPolicyTests(unittest.TestCase):
    def setUp(self):
        self.p = EgressPolicy(domains={"api.acme.com", "*.stripe.com"})

    def test_empty_policy_denies_everything(self):
        self.assertEqual(EgressPolicy().authorize("https://example.com").verdict, Verdict.BLOCK)

    def test_exact_and_wildcard(self):
        self.assertEqual(self.p.authorize("https://api.acme.com/v1/x").verdict, Verdict.ALLOW)
        self.assertEqual(self.p.authorize("https://files.stripe.com/x").verdict, Verdict.ALLOW)
        self.assertEqual(self.p.authorize("https://stripe.com/").verdict, Verdict.ALLOW)
        self.assertEqual(self.p.authorize("https://acme.com/").verdict, Verdict.BLOCK)
        self.assertEqual(self.p.authorize("https://api.acme.com.evil.com/").verdict, Verdict.BLOCK)
        self.assertEqual(self.p.authorize("https://evilstripe.com/").verdict, Verdict.BLOCK)

    def test_scheme(self):
        self.assertEqual(self.p.authorize("http://api.acme.com/").verdict, Verdict.BLOCK)
        self.assertEqual(self.p.authorize("file:///etc/passwd").verdict, Verdict.BLOCK)
        self.assertEqual(self.p.authorize("javascript:alert(1)").verdict, Verdict.BLOCK)

    def test_private_and_metadata(self):
        for url in [
            "https://127.0.0.1/",
            "https://10.0.0.5/",
            "https://192.168.1.1/",
            "https://169.254.169.254/latest/meta-data/",
            "https://[::1]/",
            "https://localhost/",
            "https://db.internal/",
            "https://metadata.google.internal/",
        ]:
            d = self.p.authorize(url)
            self.assertEqual(d.verdict, Verdict.BLOCK, msg=url)

    def test_ip_literal_even_when_allowlisted(self):
        p = EgressPolicy(domains={"8.8.8.8"})
        self.assertEqual(p.authorize("https://8.8.8.8/").verdict, Verdict.BLOCK)
        p = EgressPolicy(domains={"8.8.8.8"}, allow_ip_literals=True)
        self.assertEqual(p.authorize("https://8.8.8.8/").verdict, Verdict.ALLOW)
        self.assertEqual(p.authorize("https://10.0.0.1/").verdict, Verdict.BLOCK)

    def test_userinfo(self):
        self.assertEqual(self.p.authorize("https://user:pw@api.acme.com/").verdict, Verdict.BLOCK)
        self.assertEqual(self.p.authorize("https://api.acme.com@evil.com/").verdict, Verdict.BLOCK)

    def test_ports(self):
        p = EgressPolicy(domains={"api.acme.com"}, ports={443})
        self.assertEqual(p.authorize("https://api.acme.com:8443/").verdict, Verdict.BLOCK)
        self.assertEqual(p.authorize("https://api.acme.com:443/").verdict, Verdict.ALLOW)

    def test_secret_in_query_flags(self):
        d = self.p.authorize("https://api.acme.com/cb?token=abc")
        self.assertEqual(d.verdict, Verdict.FLAG)

    def test_garbage(self):
        for url in ["", None, "https://", "https://api.acme.com/\x00", "https://exa mple.com", "https://" + "a" * 3000]:
            self.assertEqual(self.p.authorize(url).verdict, Verdict.BLOCK, msg=repr(url))  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
