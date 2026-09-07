# Security Policy

KreGuard is a security library. We take reports seriously and want to hear about anything that weakens the guarantees described in the README.

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting a vulnerability

Please do not open a public issue for security problems.

Use GitHub's private reporting instead: go to the **Security** tab of this repository and choose **Report a vulnerability**. Include a minimal reproduction (the input text, the policy or rule set in use, and the observed vs expected decision).

You can expect an acknowledgement within 72 hours and a status update within 14 days. Once a fix is released we will credit you in the release notes unless you ask us not to.

## What counts

In scope:

- Bypasses of the permission or policy layer (enforcement)
- Inputs that crash or hang the classifier or CLI (denial of service)
- Unsafe defaults that silently allow dangerous actions

Out of scope:

- New jailbreak phrasings that slip past the pattern detector. Text defenses are advisory by design. Please open a normal issue or PR with the sample so it can be added to the rule set.
- Issues in third-party judge backends you plug into KreGuard

## Scope of guarantees

KreGuard's text-based detection is best effort and will never be complete. The enforcement layer (permissions, policy, sandboxed actions) is where hard guarantees live. If you rely on detection alone, you are relying on advice, not a wall.
