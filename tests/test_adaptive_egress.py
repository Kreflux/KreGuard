import base64
import json
import random
import tempfile
import unittest
from pathlib import Path

from kreguard import AdaptiveEgress, EgressPolicy, FilterList, Guard, ModelError, Verdict
from kreguard.adaptive.model import save_json

WORDS = "order shipping refund invoice status account billing return widget".split()


def opaque(rng, n=150):
    return base64.urlsafe_b64encode(bytes(rng.randrange(256) for _ in range(n))).decode().rstrip("=")


class Traffic:
    def __init__(self, seed=1):
        self.rng = random.Random(seed)

    def normal(self, host="api.acme.com"):
        r = self.rng
        return f"https://{host}/v1/search?q={r.choice(WORDS)}+{r.choice(WORDS)}&page={r.randint(1, 3)}"


def make(**kw):
    learner = AdaptiveEgress(**kw)
    policy = EgressPolicy(domains={"api.acme.com", "*.stripe.com"}, learner=learner)
    return policy, learner


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.policy, self.learner = make()
        self.t = Traffic()
        for _ in range(40):
            self.assertEqual(self.policy.authorize(self.t.normal()).verdict, Verdict.ALLOW)

    def test_normal_traffic_keeps_flowing(self):
        for _ in range(50):
            self.assertEqual(self.policy.authorize(self.t.normal()).verdict, Verdict.ALLOW)

    def test_smuggled_data_to_an_allowed_host_is_blocked(self):
        url = f"https://api.acme.com/v1/search?q={opaque(self.t.rng)}&page=1"
        d = self.policy.authorize(url)
        self.assertEqual(d.verdict, Verdict.BLOCK)
        self.assertEqual(d.findings[0].rule, "anomalous_request")
        self.assertIn("deviations above normal for api.acme.com", d.findings[0].detail)

    def test_a_merely_longer_honest_request_is_not_blocked(self):
        d = self.policy.authorize("https://api.acme.com/v1/search?q=where+is+my+order+number+1234+placed+last+week&page=2")
        self.assertEqual(d.verdict, Verdict.ALLOW)

    def test_an_unusual_but_plain_request_is_flagged_not_blocked(self):
        d = self.policy.authorize("https://api.acme.com/v1/search?q=" + "a" * 60 + "&page=2")
        self.assertEqual(d.verdict, Verdict.FLAG)
        self.assertIn("unusual_request", [f.rule for f in d.findings])

    def test_a_new_host_has_no_baseline_so_only_static_rules_apply(self):
        d = self.policy.authorize(f"https://pay.stripe.com/x?q={opaque(self.t.rng, 60)}")
        self.assertNotIn("anomalous_request", [f.rule for f in d.findings])
        self.assertNotIn("unusual_request", [f.rule for f in d.findings])

    def test_a_young_subdomain_borrows_its_parent_domains_baseline(self):
        pol, _ = make()
        for _ in range(30):
            pol.authorize(f"https://a.stripe.com/v1/charges?limit={self.t.rng.randint(1, 9)}")
        d = pol.authorize(f"https://b.stripe.com/v1/charges?x={opaque(self.t.rng)}")
        self.assertEqual(d.verdict, Verdict.BLOCK)

    def test_flagged_and_blocked_requests_do_not_teach_the_baseline(self):
        before = self.learner.stats()["observed"]
        self.policy.authorize("https://api.acme.com/v1/search?q=" + "a" * 60)
        self.policy.authorize(f"https://api.acme.com/v1/search?q={opaque(self.t.rng)}")
        self.assertEqual(self.learner.stats()["observed"], before)

    def test_a_slow_ramp_is_caught_before_it_gets_big(self):
        """Growing each request a little cannot drag the baseline along with it."""
        policy = EgressPolicy(domains={"api.acme.com"}, learner=AdaptiveEgress(), flag_query_secrets=False)
        t = Traffic(7)
        for _ in range(40):
            policy.authorize(t.normal())
        n, stopped = 20.0, None
        while n < 5000:
            q = "".join(t.rng.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(int(n)))
            d = policy.authorize(f"https://api.acme.com/v1/search?q={q}&page=1")
            if d.verdict is not Verdict.ALLOW:
                stopped = int(n)
                break
            n *= 1.08
        self.assertIsNotNone(stopped)
        self.assertLess(stopped, 120)

    def test_sitting_just_under_the_line_does_not_move_the_line(self):
        """Tolerated requests are allowed but do not teach, so repeating them proves nothing."""
        policy = EgressPolicy(domains={"api.acme.com"}, learner=AdaptiveEgress(), flag_query_secrets=False)
        t = Traffic(7)
        for _ in range(40):
            policy.authorize(t.normal())

        def url(n):
            return "https://api.acme.com/v1/search?q=" + "a" * n + "&page=1"

        def ceiling():
            return max(n for n in range(10, 400) if not policy.learner.assess(url(n)))

        edge = ceiling()
        for _ in range(500):
            policy.authorize(url(edge))
        self.assertEqual(ceiling(), edge)

    def test_a_human_can_say_a_flagged_request_was_fine(self):
        url = "https://api.acme.com/v1/search?q=" + "a" * 60 + "&page=2"
        self.assertEqual(self.policy.authorize(url).verdict, Verdict.FLAG)
        for _ in range(12):
            self.learner.report(url, False)
        self.assertNotIn("unusual_request", [f.rule for f in self.policy.authorize(url).findings])

    def test_the_learner_never_allows_what_the_allowlist_refuses(self):
        pol = EgressPolicy(domains={"api.acme.com"}, learner=AdaptiveEgress())
        t = Traffic()
        for _ in range(40):
            pol.authorize(t.normal())
        self.assertEqual(pol.authorize("https://other.example.com/").verdict, Verdict.BLOCK)
        pol.learner.report("https://other.example.com/", False)
        self.assertEqual(pol.authorize("https://other.example.com/").verdict, Verdict.BLOCK)

    def test_baselines_are_bounded(self):
        pol, learner = make(max_hosts=10)
        pol.domains.add("*.acme.com")
        for i in range(100):
            pol.authorize(f"https://h{i}.acme.com/x")
        self.assertLessEqual(learner.stats()["hosts_tracked"], 10)


class LearnedBlocklistTests(unittest.TestCase):
    def setUp(self):
        self.learner = AdaptiveEgress()
        self.policy = EgressPolicy(domains={"*.example.org", "api.acme.com"}, learner=self.learner)

    def test_reporting_a_host_blocks_it_and_its_subdomains_at_once(self):
        self.assertEqual(self.policy.authorize("https://collector.example.org/x").verdict, Verdict.ALLOW)
        self.learner.report("https://collector.example.org/x", True)
        for url in ("https://collector.example.org/x", "https://deep.collector.example.org/"):
            d = self.policy.authorize(url)
            self.assertEqual(d.verdict, Verdict.BLOCK, url)
            self.assertEqual(d.findings[0].rule, "learned_blocklist")
        self.assertEqual(self.policy.authorize("https://docs.example.org/").verdict, Verdict.ALLOW)

    def test_reporting_it_safe_forgives_it(self):
        self.learner.report("https://collector.example.org/", True)
        out = self.learner.report("https://deep.collector.example.org/", False)
        self.assertEqual(out["removed_rules"], ["collector.example.org"])
        self.assertEqual(self.policy.authorize("https://collector.example.org/").verdict, Verdict.ALLOW)

    def test_lookalikes_of_a_reported_host_score_high(self):
        before = self.learner.risk("https://zq-drop-box-77.example.org/")
        self.learner.report("https://zq-drop-box-1.example.org/", True)
        after = self.learner.risk("https://zq-drop-box-77.example.org/")
        self.assertGreater(after, before + 0.3)
        self.assertLess(self.learner.risk("https://api.acme.com/v1/orders"), 0.2)

    def test_the_risk_model_starts_out_knowing_the_usual_suspects(self):
        L = self.learner
        for bad in ("https://webhook.site/abc", "https://abc123.ngrok-free.dev/x", "https://webhook-collector.xyz/"):
            self.assertGreater(L.risk(bad), 0.9, bad)
        for good in ("https://api.github.com/", "https://api.stripe.com/v1/charges", "https://docs.python.org/3/"):
            self.assertLess(L.risk(good), 0.2, good)

    def test_risk_blocks_a_lookalike_that_slips_through_a_wildcard(self):
        pol = EgressPolicy(domains={"*.xyz"}, learner=self.learner)
        d = pol.authorize("https://webhook-collector.xyz/")
        self.assertEqual(d.verdict, Verdict.FLAG)  # resembles known-bad hosts, not proven
        self.assertEqual(d.findings[0].rule, "learned_risk")
        self.learner.report("https://webhook-collector.xyz/", True)  # a person confirms it
        self.assertEqual(pol.authorize("https://webhook-collector.xyz/").verdict, Verdict.BLOCK)

    def test_seed_can_leave_out_the_builtin_denylist(self):
        bare = AdaptiveEgress(seed_malicious=False)
        self.assertLess(bare.risk("https://webhook.site/abc"), 0.7)

    def test_a_filter_list_hit_teaches_the_risk_model(self):
        L = AdaptiveEgress(seed_malicious=False)
        pol = EgressPolicy(domains={"*.xyz"}, blocklist=FilterList("||zq-drop-box-1.xyz^"), learner=L)
        before = L.risk("https://zq-drop-box-2.xyz/")
        self.assertEqual(pol.authorize("https://zq-drop-box-1.xyz/").verdict, Verdict.BLOCK)
        self.assertGreater(L.risk("https://zq-drop-box-2.xyz/"), before)
        self.assertEqual(L.stats()["auto_bad"], 1)

    def test_auto_learning_from_the_filter_list_is_rate_limited(self):
        now = [0.0]
        L = AdaptiveEgress(auto_per_hour=2, clock=lambda: now[0])
        self.assertTrue(L.observe_malicious("https://a1.xyz/"))
        self.assertTrue(L.observe_malicious("https://a2.xyz/"))
        self.assertFalse(L.observe_malicious("https://a3.xyz/"))
        now[0] += 3601
        self.assertTrue(L.observe_malicious("https://a4.xyz/"))

    def test_unreadable_urls_are_rejected_not_guessed(self):
        with self.assertRaises(ValueError):
            self.learner.report("not a url", True)

    def test_guard_feedback_wiring(self):
        g = Guard(egress_policy=self.policy)
        out = g.feedback_egress("https://collector.example.org/", True)
        self.assertEqual(out["host"], "collector.example.org")
        self.assertTrue(g.authorize_egress("https://collector.example.org/").blocked)
        self.assertEqual(g.model_stats()["egress"]["learned_blocked_count"], 1)


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "egress.json"

    def test_baselines_and_learned_blocks_survive_a_restart(self):
        a = AdaptiveEgress(path=self.path)
        pol = EgressPolicy(domains={"api.acme.com", "*.example.org"}, learner=a)
        t = Traffic()
        for _ in range(40):
            pol.authorize(t.normal())
        a.report("https://bad.example.org/", True)
        a.save()
        b = AdaptiveEgress(path=self.path)
        pol2 = EgressPolicy(domains={"api.acme.com", "*.example.org"}, learner=b)
        self.assertEqual(pol2.authorize("https://bad.example.org/").verdict, Verdict.BLOCK)
        self.assertEqual(pol2.authorize(f"https://api.acme.com/v1/search?q={opaque(t.rng)}").verdict, Verdict.BLOCK)
        self.assertEqual(b.stats()["hosts_with_baseline"], a.stats()["hosts_with_baseline"])

    def test_autosave_and_failure_reporting(self):
        a = AdaptiveEgress(path=self.path, autosave_every=3)
        for i in range(3):
            a.observe(f"https://api.acme.com/x?i={i}")
        self.assertTrue(self.path.exists())
        broken = AdaptiveEgress(path=Path(self.tmp.name) / "f" / "x.json", autosave_every=1)
        Path(self.tmp.name, "f").write_text("not a directory")
        broken.observe("https://api.acme.com/x")
        self.assertIsNotNone(broken.stats()["last_save_error"])

    def test_hostile_files_are_refused(self):
        a = AdaptiveEgress(path=self.path)
        a.observe("https://api.acme.com/x")
        a.save()
        good = json.loads(self.path.read_text())["payload"]
        for mutate in (
            lambda p: p["baselines"].__setitem__("x.com", {"n": -1, "mean": {}, "var": {}}),
            lambda p: p["baselines"].__setitem__("x.com", {"n": 1, "mean": {"path_len": 1e30}, "var": {}}),
            lambda p: p["bad"].__setitem__("evil.com", ["x", "feedback"]),
            lambda p: p["bad"].__setitem__("evil.com", [1.0, "root"]),
            lambda p: p.__setitem__("baselines", []),
            lambda p: p.pop("risk"),
            lambda p: p["risk"].__setitem__("dim", 1 << 10),
        ):
            payload = json.loads(json.dumps(good))
            mutate(payload)
            save_json(self.path, "adaptive-egress", payload)
            with self.assertRaises(ModelError):
                AdaptiveEgress(path=self.path)

    def test_bad_settings_rejected(self):
        with self.assertRaises(ValueError):
            AdaptiveEgress(flag_z=10, block_z=5)
        with self.assertRaises(ValueError):
            AdaptiveEgress(min_samples=1)


if __name__ == "__main__":
    unittest.main()
