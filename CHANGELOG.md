# Changelog

## 0.3.0

- Adaptive text classifier: an online model that starts from a built-in seed, learns from feedback, remembers confirmed attacks, and teaches itself from hard evidence. Only toward blocking.
- Adaptive egress: per-destination baselines that catch data smuggled through allowed hosts, a learned destination-risk model, and a learned blocklist.
- Learning is advisory, rate limited, bounded and graded before it is learned. Model files are checksummed and validated.
- `feedback_input`, `feedback_egress`, `/v1/feedback/*`, `/v1/model`, playground teaching buttons, and `learn`, `report`, `train`, `model` commands.
- `classifier` in config now defaults to `adaptive`.

## 0.2.0

- HTTP service (`python -m kreguard serve`) with a web playground, token auth and a loopback-only default.
- JSON config files with strict validation and declarative tool argument constraints.
- Adblock-syntax filter lists as an egress denylist, plus a built-in list of exfiltration endpoints.
- JSON Lines audit log that stores hashes, not text, by default.
- `--config`, `--blocklist` and `--builtin-blocklist` on the CLI.

## 0.1.0

- Normalization, pattern scanner, classifier and judge interfaces, output scanner, tool and egress gates, CLI.
