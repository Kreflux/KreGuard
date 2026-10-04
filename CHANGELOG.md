# Changelog

## 0.2.0

- HTTP service (`python -m kreguard serve`) with a web playground, token auth and a loopback-only default.
- JSON config files with strict validation and declarative tool argument constraints.
- Adblock-syntax filter lists as an egress denylist, plus a built-in list of exfiltration endpoints.
- JSON Lines audit log that stores hashes, not text, by default.
- `--config`, `--blocklist` and `--builtin-blocklist` on the CLI.

## 0.1.0

- Normalization, pattern scanner, classifier and judge interfaces, output scanner, tool and egress gates, CLI.
