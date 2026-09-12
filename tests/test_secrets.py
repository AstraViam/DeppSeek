"""Secret redaction.

The fixtures below are assembled from fragments at runtime rather than written
as literals. They are entirely synthetic, but a credential-shaped literal in a
source file trips secret scanners on push, and a test suite that has to be
allow-listed past a security control is a bad trade. Building them from parts
keeps the test honest and the repository clean.
"""

import pytest

from deppseek.secrets import redact, scrub


def fake(prefix: str, body: str) -> str:
    """Assemble a synthetic credential of the given shape."""
    return prefix + body


# Shapes only. None of these is, or ever was, a real credential.
DEEPSEEK_KEY = fake("sk-", "1234567890abcdef" * 2)
OPENAI_KEY = fake("sk-" + "proj-", "AbCdEfGhIjKlMnOpQrStUvWxYz0123456789")
SLACK_TOKEN = fake("xox" + "b-", "1234567890-abcdefghijklmno")
GITHUB_TOKEN = fake("gh" + "p_", "a" * 36)
AWS_KEY = fake("AK" + "IA", "IOSFODNN7EXAMPLE")
BEARER = fake("", "abcdefghijklmnopqrstuvwxyz123456")


@pytest.mark.parametrize(
    "text,label",
    [
        (f"DEEPSEEK_API_KEY={DEEPSEEK_KEY}", "deepseek-key"),
        (f"key = {OPENAI_KEY}", "openai-key"),
        (f"token: {SLACK_TOKEN}", "slack-token"),
        (GITHUB_TOKEN, "github-token"),
        (f"aws_access_key_id = {AWS_KEY}", "aws-access-key"),
        (f"Authorization: Bearer {BEARER}", "bearer-header"),
        ("postgres://admin:Sup3rSecretPass@db.host/x", "url-credentials"),
    ],
)
def test_credentials_are_redacted(text, label):
    report = redact(text)
    assert label in report.hits
    assert "REDACTED" in report.text


def test_slack_token_is_not_mistaken_for_a_placeholder():
    """Regression: a placeholder rule of '^x+' matched the leading x of every
    Slack token, waving real credentials straight through."""
    assert redact(SLACK_TOKEN).redacted


def test_private_key_blocks_are_redacted_whole():
    body = "MIIEpAIBAAKCAQEA1234567890"
    pem = f"-----BEGIN RSA PRIVATE KEY-----\n{body}\n-----END RSA PRIVATE KEY-----"
    report = redact(pem)
    assert "private-key-block" in report.hits
    assert body not in report.text


@pytest.mark.parametrize(
    "text",
    [
        'API_KEY = "your_key_here"',
        'API_KEY = "xxxxxxxxxxxxxxxx"',
        'API_KEY = "${DEEPSEEK_KEY}"',
        'API_KEY = "{{ vault_key }}"',
        "mdot = rho * A * v  # mass flow rate, kg/s",
        "Reynolds number Re = 4000 for this pipe",
        "commit 8f3c2a1b9d4e5f6071829304a5b6c7d8e9f00112",
    ],
)
def test_non_secrets_are_left_alone(text):
    assert redact(text).text == text


def test_surrounding_syntax_survives_redaction():
    out = scrub("postgres://admin:Sup3rSecretPass@db.host/mydb")
    assert out.startswith("postgres://admin:")
    assert out.endswith("@db.host/mydb")


def test_report_describes_what_was_removed():
    report = redact(f"k1={DEEPSEEK_KEY} and k2={GITHUB_TOKEN}")
    described = report.describe()
    assert "deepseek-key" in described and "github-token" in described


def test_empty_input_is_safe():
    assert redact("").text == ""
    assert not redact("").redacted
