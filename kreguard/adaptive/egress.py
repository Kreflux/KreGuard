"""Adaptive egress: learn what normal outbound traffic looks like.

A static allowlist says where the app may talk. It says nothing about what a
request carries. Data smuggled out through an allowed destination, one long
opaque query value at a time, passes every static rule. This module learns
the difference, in three ways.

1. Per-destination baselines. For every host the app has talked to, it keeps
   a running picture of a normal request: path length, query length, number
   of parameters, longest opaque token, entropy. A request far outside that
   picture is flagged, and blocked when it also looks like smuggled data
   (a long, high-entropy opaque value). Only requests the policy allowed teach
   the baseline, and each sample is winsorized, so an attacker cannot drag the
   baseline upward with a slow ramp of bigger and bigger requests.

2. A learned destination-risk model. A small online classifier over hostname
   features. It is seeded with the built-in denylist and well-known good
   hosts, then learns from feedback and from the filter list. Told that one
   host is malicious, it also scores that host's lookalikes high.

3. A learned blocklist. A host reported as malicious is blocked immediately,
   along with its subdomains, no list edit and no redeploy. Reporting it safe
   removes it.

Like everything adaptive in KreGuard this sits on top of the static policy. It
can add flags and blocks to an allowed request. It never allows one the
allowlist refused, and it never widens an exception.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Union
from urllib.parse import urlsplit

from ..filterlist import DEFAULT_RULES
from ..verdict import Finding, Verdict
from .features import hash_features, request_shape, url_features
from .model import ModelError, OnlineLogReg, load_json, save_json

_KIND = "adaptive-egress"

# A request feature cannot be judged against a baseline tighter than this.
_STD_FLOOR = {
    "path_len": 12.0,
    "query_len": 24.0,
    "n_params": 1.5,
    "opaque_len": 10.0,
    "entropy": 0.6,
    "path_depth": 1.0,
}
_FEATURES = tuple(_STD_FLOOR)
# Features that carry data out; the only ones that can justify a block.
_CARRIERS = ("query_len", "opaque_len", "path_len")

_BENIGN_HOSTS = """
github.com api.github.com raw.githubusercontent.com gitlab.com bitbucket.org pypi.org files.pythonhosted.org
python.org docs.python.org npmjs.com registry.npmjs.org crates.io rubygems.org golang.org pkg.go.dev
wikipedia.org en.wikipedia.org wikimedia.org stackoverflow.com stackexchange.com mozilla.org developer.mozilla.org
google.com www.google.com googleapis.com storage.googleapis.com maps.googleapis.com gstatic.com youtube.com
microsoft.com graph.microsoft.com login.microsoftonline.com azure.com windows.net office.com live.com
amazonaws.com s3.amazonaws.com aws.amazon.com amazon.com apple.com icloud.com cloudflare.com
stripe.com api.stripe.com paypal.com api.paypal.com twilio.com api.twilio.com sendgrid.com api.sendgrid.com
slack.com api.slack.com discord.com notion.so api.notion.com atlassian.net atlassian.com trello.com
openai.com api.openai.com anthropic.com api.anthropic.com huggingface.co arxiv.org nature.com
salesforce.com force.com zendesk.com hubspot.com shopify.com myshopify.com intercom.io
datadoghq.com sentry.io newrelic.com grafana.com elastic.co mongodb.com redis.io postgresql.org
linkedin.com twitter.com x.com facebook.com instagram.com reddit.com medium.com nytimes.com bbc.co.uk
acme.com api.acme.com docs.acme.com support.acme.com
""".split()


def _bad_hosts() -> List[str]:
    out = []
    for line in DEFAULT_RULES.splitlines():
        line = line.strip()
        if line.startswith("||") and line.endswith("^"):
            out.append(line[2:-1])
    return out


def host_of(url: str) -> str:
    try:
        return (urlsplit(url.strip()).hostname or "").lower().rstrip(".")
    except ValueError:
        return ""


def parent_of(host: str) -> str:
    labels = host.split(".")
    return ".".join(labels[-2:]) if len(labels) > 2 else host


class _Baseline:
    __slots__ = ("n", "mean", "var")

    def __init__(self) -> None:
        self.n = 0
        self.mean = {k: 0.0 for k in _FEATURES}
        self.var = {k: 0.0 for k in _FEATURES}

    def std(self, k: str) -> float:
        return max(self.var[k] ** 0.5, _STD_FLOOR[k])

    def z(self, shape: Dict[str, float]) -> Dict[str, float]:
        return {k: (shape[k] - self.mean[k]) / self.std(k) for k in _FEATURES}

    def update(self, shape: Dict[str, float], alpha: float, winsorize: bool) -> None:
        """Exponentially weighted mean and variance, true running average at first.

        With ``winsorize`` (every automatic update) a sample is clamped to one
        standard deviation of the current mean. That bounds how far any single
        request can move the baseline, and it means the spread can only
        shrink, never grow, once the baseline has warmed up. An attacker
        cannot widen what counts as normal by sending bigger requests; only a
        person confirming a request is fine (``winsorize=False``) can.
        """
        a = max(1.0 / (self.n + 1), alpha)
        for k in _FEATURES:
            x = shape[k]
            if winsorize and self.n >= 5:
                sd = self.std(k)
                x = max(self.mean[k] - sd, min(x, self.mean[k] + sd))
            delta = x - self.mean[k]
            self.mean[k] += a * delta
            self.var[k] = (1.0 - a) * (self.var[k] + a * delta * delta)
        self.n += 1


class AdaptiveEgress:
    def __init__(
        self,
        *,
        path: Union[str, Path, None] = None,
        autosave_every: int = 0,
        min_samples: int = 20,
        flag_z: float = 4.0,
        block_z: float = 10.0,
        learn_z: float = 2.0,
        alpha: float = 0.05,
        risk_flag: float = 0.7,
        risk_block: float = 0.97,
        max_hosts: int = 5000,
        max_learned_hosts: int = 20000,
        auto_per_hour: int = 500,
        seed: bool = True,
        seed_malicious: bool = True,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not (0 < flag_z < block_z) or min_samples < 5:
            raise ValueError("need 0 < flag_z < block_z and min_samples >= 5")
        self.path = Path(path) if path is not None else None
        self.autosave_every = autosave_every
        self.min_samples = min_samples
        self.flag_z = flag_z
        self.block_z = block_z
        self.learn_z = learn_z
        self.alpha = alpha
        self.risk_flag = risk_flag
        self.risk_block = risk_block
        self.max_hosts = max_hosts
        self.max_learned_hosts = max_learned_hosts
        self.auto_per_hour = auto_per_hour
        self._clock = clock
        self._lock = threading.RLock()
        self._baselines: "OrderedDict[str, _Baseline]" = OrderedDict()
        self._bad: Dict[str, List[Any]] = {}   # host -> [timestamp, source]
        self._risk = OnlineLogReg(dim=1 << 16, lr=0.5)
        self._benign_seen: Deque[str] = deque(maxlen=4096)
        self._benign_seen_set: set = set()
        self._auto_times: Deque[float] = deque()
        self.counts: Dict[str, int] = {
            "observed": 0, "flagged_anomaly": 0, "blocked_anomaly": 0,
            "blocked_learned": 0, "feedback_bad": 0, "feedback_safe": 0, "auto_bad": 0,
        }
        self._since_save = 0
        self.last_save_error: Optional[str] = None

        if self.path is not None and self.path.exists():
            self._load(self.path)
        elif seed:
            self._seed(seed_malicious)

    # Assessment

    def assess(self, url: str) -> List[Finding]:
        """Findings for a URL the static policy was about to allow."""
        host = host_of(url)
        if not host:
            return []
        findings: List[Finding] = []
        with self._lock:
            bad = self._bad_match(host)
            if bad is not None:
                self.counts["blocked_learned"] += 1
                return [Finding("egress", "learned_blocklist", Verdict.BLOCK, f"{bad} was reported malicious", 1.0)]

            p = self._risk.predict(hash_features(url_features(url), self._risk.dim))
            if p >= self.risk_block:
                findings.append(Finding("egress", "learned_risk", Verdict.BLOCK, f"destination resembles known-bad hosts (p={p:.2f})", p))
            elif p >= self.risk_flag:
                findings.append(Finding("egress", "learned_risk", Verdict.FLAG, f"destination resembles known-bad hosts (p={p:.2f})", p))

            base = self._baseline_for(host)
            if base is not None:
                shape = request_shape(url)
                z = base.z(shape)
                worst_key = max(_FEATURES, key=lambda k: z[k])
                worst = z[worst_key]
                carrier_z = max(z[k] for k in _CARRIERS)
                smuggling = shape["opaque_len"] >= 24 and shape["entropy"] >= 3.5
                if carrier_z >= self.block_z and smuggling:
                    self.counts["blocked_anomaly"] += 1
                    findings.append(Finding(
                        "egress", "anomalous_request", Verdict.BLOCK,
                        f"{worst_key} is {worst:.1f} deviations above normal for {host} and carries a long opaque value", min(1.0, worst / (self.block_z * 1.5)),
                    ))
                elif worst >= self.flag_z:
                    self.counts["flagged_anomaly"] += 1
                    findings.append(Finding(
                        "egress", "unusual_request", Verdict.FLAG,
                        f"{worst_key} is {worst:.1f} deviations above normal for {host}", min(0.9, worst / (self.block_z * 1.5)),
                    ))
        return findings

    def observe(self, url: str) -> None:
        """A request the policy allowed in full. It teaches what normal is."""
        host = host_of(url)
        if not host:
            return
        with self._lock:
            shape = request_shape(url)
            for key in {host, parent_of(host)}:
                base = self._baseline_get(key, create=True)
                # A mature baseline only learns from typical requests. One that
                # is merely tolerated (unusual but under the flag line) is
                # allowed through without teaching it that this is normal.
                if base.n >= self.min_samples and max(base.z(shape).values()) >= self.learn_z:
                    continue
                base.update(shape, self.alpha, winsorize=True)
            self.counts["observed"] += 1
            if host not in self._benign_seen_set:
                if len(self._benign_seen) == self._benign_seen.maxlen:
                    self._benign_seen_set.discard(self._benign_seen[0])
                self._benign_seen.append(host)
                self._benign_seen_set.add(host)
                # The allowlist is the operator's own statement that this host is fine.
                self._fit_risk(url, 0, steps=2)
            self._tick()

    def observe_malicious(self, url: str) -> bool:
        """A hard block from the static policy (a filter-list hit) teaches the risk model."""
        with self._lock:
            now = self._clock()
            while self._auto_times and now - self._auto_times[0] > 3600.0:
                self._auto_times.popleft()
            if len(self._auto_times) >= self.auto_per_hour:
                return False
            self._auto_times.append(now)
            self.counts["auto_bad"] += 1
            self._fit_risk(url, 1, steps=2)
            self._tick()
            return True

    # Feedback

    def report(self, url: str, malicious: bool) -> Dict[str, Any]:
        """Ground truth from a human about a destination."""
        host = host_of(url)
        if not host:
            raise ValueError("cannot read a host from that URL")
        with self._lock:
            out: Dict[str, Any] = {"host": host, "malicious": malicious}
            if malicious:
                if len(self._bad) >= self.max_learned_hosts and host not in self._bad:
                    oldest = min(self._bad, key=lambda h: self._bad[h][0])
                    del self._bad[oldest]
                self._bad[host] = [round(self._clock(), 3), "feedback"]
                self.counts["feedback_bad"] += 1
                out["p_after"] = self._fit_risk(url, 1, steps=8)
            else:
                removed = self._forget(host)
                self.counts["feedback_safe"] += 1
                out["removed_rules"] = removed
                out["p_after"] = self._fit_risk(url, 0, steps=8)
                # A human saying this request is fine is how the baseline adapts to change.
                shape = request_shape(url)
                for key in {host, parent_of(host)}:
                    self._baseline_get(key, create=True).update(shape, self.alpha * 2, winsorize=False)
            self._tick()
            return out

    def _forget(self, host: str) -> List[str]:
        removed = []
        for h in list(self._bad):
            if host == h or host.endswith("." + h):
                removed.append(h)
                del self._bad[h]
        return removed

    def _bad_match(self, host: str) -> Optional[str]:
        labels = host.split(".")
        for i in range(len(labels)):
            cand = ".".join(labels[i:])
            if cand in self._bad:
                return cand
        return None

    # Internals

    def _fit_risk(self, url: str, target: int, steps: int) -> float:
        feats = hash_features(url_features(url), self._risk.dim)
        p = self._risk.predict(feats)
        for _ in range(steps):
            if abs(p - target) <= 0.1:
                break
            self._risk.update(feats, target)
            p = self._risk.predict(feats)
        return round(p, 4)

    def _baseline_get(self, host: str, create: bool) -> Optional[_Baseline]:
        base = self._baselines.get(host)
        if base is not None:
            self._baselines.move_to_end(host)
            return base
        if not create:
            return None
        base = _Baseline()
        self._baselines[host] = base
        while len(self._baselines) > self.max_hosts:
            self._baselines.popitem(last=False)
        return base

    def _baseline_for(self, host: str) -> Optional[_Baseline]:
        """The host's own baseline once it is mature, else its parent domain's."""
        own = self._baselines.get(host)
        if own is not None and own.n >= self.min_samples:
            return own
        parent = self._baselines.get(parent_of(host))
        if parent is not None and parent.n >= self.min_samples:
            return parent
        return None

    def _seed(self, malicious: bool = True) -> None:
        bad = _bad_hosts() if malicious else []
        pos = [f"https://{h}/" for h in bad] + [f"https://{lab}.{h}/x?id=1" for h in bad for lab in ("a1b2c3d4", "demo-7f3a")]
        neg = [f"https://{h}/" for h in _BENIGN_HOSTS] + [f"https://www.{h}/docs/page?lang=en" for h in _BENIGN_HOSTS[:40]]
        data = [(hash_features(url_features(u), self._risk.dim), 1) for u in pos] + [(hash_features(url_features(u), self._risk.dim), 0) for u in neg]
        order = []
        n = max(len(pos), len(neg))
        pf = [d for d in data if d[1] == 1]
        nf = [d for d in data if d[1] == 0]
        for i in range(n):
            if pf:
                order.append(pf[i % len(pf)])
            order.append(nf[(i * len(nf)) // n])
        for _ in range(6):
            for feats, y in order:
                self._risk.update(feats, y)

    def _tick(self) -> None:
        self._since_save += 1
        if self.path is not None and self.autosave_every and self._since_save >= self.autosave_every:
            try:
                self.save()
                self.last_save_error = None
            except (OSError, ValueError) as exc:
                self.last_save_error = f"{type(exc).__name__}: {exc}"

    # Introspection

    def risk(self, url: str) -> float:
        return self._risk.predict(hash_features(url_features(url), self._risk.dim))

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            mature = sum(1 for b in self._baselines.values() if b.n >= self.min_samples)
            return {
                "name": "adaptive-egress",
                "hosts_tracked": len(self._baselines),
                "hosts_with_baseline": mature,
                "learned_blocked_hosts": sorted(self._bad)[:200],
                "learned_blocked_count": len(self._bad),
                "risk_model_updates": self._risk.updates,
                **self.counts,
                "persistent": self.path is not None,
                "last_save_error": self.last_save_error,
            }

    # Persistence

    def save(self, path: Union[str, Path, None] = None) -> None:
        target = Path(path) if path is not None else self.path
        if target is None:
            raise ValueError("no path to save to")
        with self._lock:
            payload = {
                "baselines": {h: {"n": b.n, "mean": b.mean, "var": b.var} for h, b in self._baselines.items()},
                "bad": self._bad,
                "risk": self._risk.to_dict(),
                "benign_seen": list(self._benign_seen),
                "counts": self.counts,
            }
            save_json(target, _KIND, payload)
            self._since_save = 0

    def _load(self, path: Path) -> None:
        p = load_json(path, _KIND)
        try:
            self._risk = OnlineLogReg.from_dict(p["risk"])
            if self._risk.dim != 1 << 16:
                raise ModelError("risk model has the wrong dimension")
            self._baselines.clear()
            base_in = p["baselines"]
            if not isinstance(base_in, dict) or len(base_in) > self.max_hosts * 2:
                raise ModelError("baselines are malformed")
            for host, b in base_in.items():
                if not isinstance(host, str) or not isinstance(b, dict):
                    raise ModelError("baseline entry is malformed")
                base = _Baseline()
                n = b["n"]
                if isinstance(n, bool) or not isinstance(n, int) or n < 0:
                    raise ModelError("baseline count out of range")
                base.n = n
                for k in _FEATURES:
                    for src, dst in ((b["mean"], base.mean), (b["var"], base.var)):
                        v = src[k]
                        if isinstance(v, bool) or not isinstance(v, (int, float)) or not (-1e9 < v < 1e9) or v != v:
                            raise ModelError("baseline value out of range")
                        dst[k] = float(v)
                    if base.var[k] < 0:
                        raise ModelError("baseline variance is negative")
                self._baselines[host] = base
            bad = p["bad"]
            if not isinstance(bad, dict) or len(bad) > self.max_learned_hosts * 2:
                raise ModelError("learned hosts are malformed")
            self._bad = {}
            for h, rec in bad.items():
                if not (isinstance(h, str) and isinstance(rec, list) and len(rec) == 2 and rec[1] in ("feedback", "auto")
                        and isinstance(rec[0], (int, float)) and not isinstance(rec[0], bool)):
                    raise ModelError("learned host entry is malformed")
                self._bad[h] = [float(rec[0]), rec[1]]
            seen = p.get("benign_seen", [])
            if not isinstance(seen, list) or not all(isinstance(x, str) for x in seen):
                raise ModelError("benign_seen is malformed")
            self._benign_seen.clear(); self._benign_seen_set.clear()
            for x in seen[-self._benign_seen.maxlen:]:
                self._benign_seen.append(x)
                self._benign_seen_set.add(x)
            counts = p["counts"]
            if not isinstance(counts, dict):
                raise ModelError("counts are malformed")
            for k in self.counts:
                v = counts.get(k, 0)
                if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                    raise ModelError("count out of range")
                self.counts[k] = v
        except (KeyError, TypeError) as exc:
            raise ModelError(f"egress model file is malformed: {exc}") from exc
