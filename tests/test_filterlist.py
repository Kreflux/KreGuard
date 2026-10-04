import unittest

from kreguard.filterlist import DEFAULT_RULES, FilterList, normalize_url_for_matching
from kreguard.permissions import EgressPolicy
from kreguard.verdict import Verdict


class FilterListTests(unittest.TestCase):
    def blocked(self, rules, url, **kw):
        return FilterList(rules, **kw).match(url).blocked

    def test_domain_anchor_covers_subdomains_not_lookalikes(self):
        fl = FilterList("||evil.com^")
        self.assertTrue(fl.match("https://evil.com/").blocked)
        self.assertTrue(fl.match("https://a.b.evil.com/x?y=1").blocked)
        self.assertFalse(fl.match("https://notevil.com/").blocked)
        self.assertFalse(fl.match("https://evil.com.good.org/").blocked)

    def test_userinfo_cannot_hide_the_real_host(self):
        self.assertTrue(self.blocked("||evil.com^", "https://good.com@evil.com/"))
        self.assertFalse(self.blocked("||good.com^", "https://good.com@evil.com/"))

    def test_exception_overrides_block(self):
        fl = FilterList("||evil.com^\n@@||evil.com/ok^")
        self.assertFalse(fl.match("https://evil.com/ok").blocked)
        self.assertEqual(fl.match("https://evil.com/ok").excepted_by, "@@||evil.com/ok^")
        self.assertTrue(fl.match("https://evil.com/okay").blocked)

    def test_important_beats_exception(self):
        fl = FilterList("||evil.com^$important\n@@||evil.com^")
        self.assertTrue(fl.match("https://evil.com/").blocked)

    def test_exception_with_unsupported_option_is_dropped(self):
        fl = FilterList("||evil.com^\n@@||evil.com^$script")
        self.assertTrue(fl.match("https://evil.com/").blocked)
        self.assertEqual(fl.stats.exception_rules, 0)

    def test_block_rule_with_unsupported_option_still_blocks(self):
        fl = FilterList("||evil.com^$third-party,script")
        self.assertTrue(fl.match("https://evil.com/").blocked)
        self.assertEqual(fl.stats.approximated, 1)

    def test_path_and_wildcard_patterns(self):
        fl = FilterList("||x.io/ads/*\n|https://exact.example/p|\nbanner*.gif")
        self.assertTrue(fl.match("https://x.io/ads/1").blocked)
        self.assertFalse(fl.match("https://x.io/other").blocked)
        self.assertTrue(fl.match("https://exact.example/p").blocked)
        self.assertFalse(fl.match("https://exact.example/p/more").blocked)
        self.assertTrue(fl.match("https://cdn.example/banner_top.gif").blocked)

    def test_hosts_file_is_exact_host_only(self):
        fl = FilterList("127.0.0.1 localhost\n0.0.0.0 tracker.net")
        self.assertTrue(fl.match("https://tracker.net/").blocked)
        self.assertFalse(fl.match("https://sub.tracker.net/").blocked)
        self.assertEqual(fl.stats.block_rules, 1)

    def test_bare_domain_covers_subdomains(self):
        fl = FilterList("bad.org")
        self.assertTrue(fl.match("https://sub.bad.org/").blocked)

    def test_comments_headers_and_cosmetic_rules_are_skipped(self):
        fl = FilterList("[Adblock Plus 2.0]\n! note\n# note\n##.ad\nexample.com##.banner\n")
        self.assertEqual(len(fl), 0)

    def test_regex_ignored_unless_enabled(self):
        self.assertFalse(self.blocked("/evil\\.com/", "https://evil.com/"))
        self.assertTrue(self.blocked("/evil\\.com/", "https://evil.com/", allow_regex=True))

    def test_invalid_regex_does_not_raise(self):
        fl = FilterList("/(unclosed/", allow_regex=True)
        self.assertEqual(len(fl), 0)

    def test_case_insensitive_by_default(self):
        self.assertTrue(self.blocked("||evil.com/Track^", "https://evil.com/track"))
        self.assertFalse(self.blocked("||evil.com/Track^$match-case", "https://evil.com/track"))

    def test_unparseable_url_does_not_match_or_raise(self):
        self.assertFalse(FilterList("||evil.com^").match("not a url").blocked)
        self.assertIsNone(normalize_url_for_matching("://"))

    def test_large_domain_list_stays_indexed(self):
        text = "\n".join(f"||d{i}.example^" for i in range(20000))
        fl = FilterList(text)
        self.assertEqual(len(fl), 20000)
        self.assertTrue(fl.match("https://x.d19999.example/").blocked)
        self.assertFalse(fl.match("https://d20000.example/").blocked)

    def test_builtin_list_catches_request_catchers(self):
        fl = FilterList.builtin()
        for url in ("https://webhook.site/abc", "https://x.ngrok-free.app/", "https://pastebin.com/raw/x"):
            self.assertTrue(fl.match(url).blocked, url)
        self.assertFalse(fl.match("https://api.github.com/").blocked)
        self.assertIn("webhook.site", DEFAULT_RULES)


class EgressBlocklistTests(unittest.TestCase):
    def test_blocklist_wins_over_allowlist(self):
        policy = EgressPolicy(domains={"*.example.com"}, blocklist=FilterList("||bad.example.com^"))
        self.assertEqual(policy.authorize("https://ok.example.com/").verdict, Verdict.ALLOW)
        d = policy.authorize("https://bad.example.com/")
        self.assertEqual(d.verdict, Verdict.BLOCK)
        self.assertEqual(d.findings[0].rule, "filter_list")

    def test_exception_in_list_can_lift_a_list_block(self):
        policy = EgressPolicy(domains={"*.example.com"}, blocklist=FilterList("||example.com^\n@@||docs.example.com^"))
        self.assertEqual(policy.authorize("https://docs.example.com/").verdict, Verdict.ALLOW)
        self.assertEqual(policy.authorize("https://www.example.com/").verdict, Verdict.BLOCK)

    def test_blocklist_does_not_open_anything_the_allowlist_closed(self):
        policy = EgressPolicy(domains=set(), blocklist=FilterList("||bad.com^"))
        self.assertEqual(policy.authorize("https://fine.com/").verdict, Verdict.BLOCK)


if __name__ == "__main__":
    unittest.main()
