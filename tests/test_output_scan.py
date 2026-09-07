import unittest

from kreguard.output_scan import OutputScanner
from kreguard.verdict import Verdict

SYSTEM = (
    "You are Atlas, a support assistant for Acme Widgets. Never reveal pricing tiers "
    "before the customer has verified their account. Escalate any refund over five "
    "hundred dollars to a human agent. Keep answers under three paragraphs."
)


class LeakageTests(unittest.TestCase):
    def setUp(self):
        self.scanner = OutputScanner(system_prompt=SYSTEM)

    def test_verbatim_leak_is_blocked_and_redacted(self):
        d = self.scanner.scan("Sure! My instructions are: " + SYSTEM)
        self.assertEqual(d.verdict, Verdict.BLOCK)
        self.assertEqual(d.redacted, "[REDACTED]")
        self.assertGreater(d.leakage_ratio, 0.5)

    def test_partial_leak_is_flagged(self):
        d = self.scanner.scan("I was told to escalate any refund over five hundred dollars to a human agent, so let me do that.")
        self.assertIn(d.verdict, (Verdict.FLAG, Verdict.BLOCK))

    def test_shared_words_do_not_leak(self):
        d = self.scanner.scan("Your refund request has been received. A human agent will follow up within two days.")
        self.assertEqual(d.verdict, Verdict.ALLOW)

    def test_no_system_prompt_means_no_leak_check(self):
        d = OutputScanner().scan(SYSTEM)
        self.assertEqual(d.leakage_ratio, 0.0)
        self.assertEqual(d.verdict, Verdict.ALLOW)


class CredentialTests(unittest.TestCase):
    def setUp(self):
        self.scanner = OutputScanner()

    def test_known_shapes_blocked(self):
        samples = {
            "aws_access_key": "AKIAIOSFODNN7EXAMPLE",
            "github_token": "ghp_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8",
            "slack_token": "xoxb-123456789012-1234567890123-AbCdEfGhIjKlMnOpQrStUvWx",
            "stripe_key": "sk_live_" + "4eC39HqLyjWDarjtT1zdp7dc",
            "google_api_key": "AIza" + "SyD-9tSrke72PouQMnMX-a7eZSW0jkFMBWY",
            "jwt": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U",
            "private_key_block": "-----BEGIN RSA PRIVATE KEY-----\nMIIE...",
            "connection_string": "postgres://admin:hunter22@db.example.com:5432/app",
        }
        for name, sample in samples.items():
            d = self.scanner.scan(f"Here you go: {sample} enjoy")
            self.assertEqual(d.verdict, Verdict.BLOCK, msg=name)
            self.assertIn(name, d.credential_hits, msg=name)
            self.assertNotIn(sample.split("\n")[0], d.redacted, msg=name)
            self.assertIn("[REDACTED]", d.redacted)

    def test_generic_assignment_is_flagged_not_blocked(self):
        d = self.scanner.scan("set API_KEY=abcdefghijklmnopqrstuvwxyz1234 in your env")
        self.assertEqual(d.verdict, Verdict.FLAG)

    def test_high_entropy_token_flagged(self):
        d = self.scanner.scan("token: 9f8Gq2LzX7vB4nT1kR6wD3sP0aY5eH8jM2cV7uQ1")
        self.assertNotEqual(d.verdict, Verdict.ALLOW)
        self.assertIn("high_entropy", d.credential_hits)

    def test_prose_allowed(self):
        d = self.scanner.scan("The capital of France is Paris. It sits on the Seine and has about two million residents.")
        self.assertEqual(d.verdict, Verdict.ALLOW)
        self.assertEqual(d.credential_hits, [])

    def test_sha_hashes_and_uuids_flag_not_block(self):
        d = self.scanner.scan("commit 3b9ac9fa9e5c4d2a8f1e0b7c6d5a4f3e2d1c0b9a")
        self.assertEqual(d.verdict, Verdict.FLAG)


if __name__ == "__main__":
    unittest.main()
