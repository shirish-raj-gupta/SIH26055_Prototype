"""Credential loading and status reporting, without ever printing a secret.

Three rules this module exists to enforce:

1. **Secrets live outside the tree.** They come from the process environment, a
   gitignored ``.env``, or the provider's own config file. Nothing here writes a
   credential to disk inside the repository.
2. **Values are never returned or logged, only presence.**
   :func:`credential_status` reports *whether* something is configured and shows
   a fingerprint, never the token. A credential that reaches a notebook output
   is a credential that has been published.
3. **Absence is normal.** The simulator, every scheduler, the benchmark and the
   tests run with no credentials at all. Credentials are needed only to publish
   or download, and the dataset regenerates from seeds when it cannot be
   downloaded.

The ``.env`` parser is deliberately dependency-free: adding ``python-dotenv``
for twenty lines would put a package between the user and their own secrets.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "CredentialStatus",
    "credential_status",
    "fingerprint",
    "load_dotenv",
    "require",
]

#: Environment variables the project reads. Documented in ``.env.example``.
KNOWN_KEYS: tuple[str, ...] = (
    "KAGGLE_API_TOKEN",
    "KAGGLE_USERNAME",
    "KAGGLE_KEY",
    "SMARTSCAN_DATASET_SLUG",
    "SMARTSCAN_MODELS_SLUG",
    "HF_TOKEN",
    "HUGGINGFACE_TOKEN",
    "SMARTSCAN_DATA",
    "SMARTSCAN_CACHE",
)

#: Values that mean "the template was copied but not filled in".
_PLACEHOLDERS: frozenset[str] = frozenset(
    {
        "",
        "your-kaggle-username",
        "0000000000000000000000000000000000",
        "hf_your_read_token_here",
        "changeme",
        "xxx",
    }
)


def fingerprint(value: str | None) -> str:
    """Return a short, non-reversible fingerprint of a secret.

    Lets a user confirm *which* credential is loaded -- and that a rotation took
    effect -- without the value ever appearing in a log or a notebook.

    Args:
        value: The secret, or ``None``.

    Returns:
        Eight hex characters, or ``'-'`` when unset.
    """
    if not value:
        return "-"
    return hashlib.blake2b(value.encode(), digest_size=4).hexdigest()


def load_dotenv(path: str | Path = ".env", override: bool = False) -> dict[str, str]:
    """Load ``KEY=value`` pairs from a ``.env`` file into the environment.

    Supports comments, blank lines, ``export KEY=value``, and single- or
    double-quoted values. Placeholder values from the template are skipped, so a
    half-filled ``.env`` fails the same way an empty one does rather than
    authenticating as ``your-kaggle-username``.

    Args:
        path: Path to the file. Missing files are not an error.
        override: Replace variables already set in the environment. Defaults to
            ``False`` so a real shell export always wins over a stale file.

    Returns:
        Mapping of the keys that were actually applied, to their fingerprints.
    """
    p = Path(path)
    if not p.is_file():
        return {}

    applied: dict[str, str] = {}
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key or value in _PLACEHOLDERS:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
            applied[key] = fingerprint(value)
    return applied


@dataclass(frozen=True)
class CredentialStatus:
    """What is configured, and what it enables.

    Attributes:
        kaggle: Whether a usable Kaggle username and key are present.
        kaggle_source: Where they came from.
        kaggle_user: The username (not a secret).
        kaggle_key_fingerprint: Fingerprint of the key, never the key.
        huggingface: Whether an HF token is present.
        hf_source: Where it came from.
        hf_token_fingerprint: Fingerprint of the token.
        lightning: Whether Lightning AI credentials are present.
        lightning_source: Where they came from.
        lightning_user: The Lightning user id (not a secret).
        lightning_key_fingerprint: Fingerprint of the key, never the key.
        dotenv_loaded: Keys applied from ``.env``.
    """

    kaggle: bool
    kaggle_source: str
    kaggle_user: str
    kaggle_key_fingerprint: str
    huggingface: bool
    hf_source: str
    hf_token_fingerprint: str
    lightning: bool = False
    lightning_source: str = "none"
    lightning_user: str = ""
    lightning_key_fingerprint: str = "-"
    dotenv_loaded: tuple[str, ...] = ()

    def report(self) -> str:
        """Return a human-readable status block containing no secrets."""
        lines = [
            "SmartScan credential status",
            "",
            f"  Kaggle        {'configured' if self.kaggle else 'NOT configured'}"
            f"   source={self.kaggle_source}",
            f"    username    {self.kaggle_user or '-'}",
            f"    key         fingerprint {self.kaggle_key_fingerprint} (value never shown)",
            f"  Hugging Face  {'configured' if self.huggingface else 'NOT configured'}"
            f"   source={self.hf_source}",
            f"    token       fingerprint {self.hf_token_fingerprint} (value never shown)",
        ]
        if self.dotenv_loaded:
            lines += ["", f"  loaded from .env: {', '.join(self.dotenv_loaded)}"]
        lines += [
            "",
            "  Neither is required. The simulator, all nine schedulers, the",
            "  benchmark and the tests run with no credentials; the dataset",
            "  regenerates from seeds when Kaggle is unreachable.",
        ]
        if not self.kaggle:
            lines += [
                "",
                "  To enable Kaggle, any ONE of these works with CLI 2.x:",
                "    kaggle auth login                       (OAuth, nothing to store)",
                "    export KAGGLE_API_TOKEN=KGAT_...        (single token)",
                "    echo KGAT_... > ~/.kaggle/access_token  (single token, file)",
                "    KAGGLE_USERNAME + KAGGLE_KEY in .env    (legacy pair; kagglehub needs this)",
            ]
        return "\n".join(lines)


def credential_status(dotenv: str | Path | None = ".env") -> CredentialStatus:
    """Report which credentials are available, without exposing any of them.

    Args:
        dotenv: ``.env`` path to load first, or ``None`` to skip.

    Returns:
        A :class:`CredentialStatus`.
    """
    applied = load_dotenv(dotenv) if dotenv else {}

    user = os.environ.get("KAGGLE_USERNAME", "")
    key = os.environ.get("KAGGLE_KEY", "")
    kaggle_source = "environment"
    if not (user and key):
        cfg = Path.home() / ".kaggle" / "kaggle.json"
        if cfg.is_file():
            try:
                import json

                blob = json.loads(cfg.read_text(encoding="utf-8"))
                user, key = blob.get("username", ""), blob.get("key", "")
                kaggle_source = str(cfg)
            except (OSError, ValueError):
                kaggle_source = f"{cfg} (unreadable)"
        else:
            kaggle_source = "none"

    # Kaggle CLI 2.x additionally accepts a single API token, either in
    # KAGGLE_API_TOKEN or in ~/.kaggle/access_token, and an OAuth login cached
    # by `kaggle auth login`. Any of them is sufficient; we report which.
    api_token = os.environ.get("KAGGLE_API_TOKEN", "")
    if api_token and api_token not in _PLACEHOLDERS:
        user, key, kaggle_source = user or "(api-token)", api_token, "KAGGLE_API_TOKEN"
    elif not (user and key):
        token_file = Path.home() / ".kaggle" / "access_token"
        if token_file.is_file():
            try:
                candidate = token_file.read_text(encoding="utf-8").strip()
                if candidate and candidate not in _PLACEHOLDERS:
                    user, key, kaggle_source = "(api-token)", candidate, str(token_file)
            except OSError:
                pass
    if not (user and key):
        oauth = Path.home() / ".kaggle" / "oauth.json"
        if oauth.is_file():
            user, key, kaggle_source = "(oauth)", "oauth", str(oauth)

    hf = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or ""
    hf_source = "environment" if hf else "none"
    if not hf:
        cached = Path.home() / ".cache" / "huggingface" / "token"
        if cached.is_file():
            hf = cached.read_text(encoding="utf-8").strip()
            hf_source = str(cached)

    # Lightning AI. The SDK reads LIGHTNING_USER_ID and LIGHTNING_API_KEY from
    # the environment, or a credentials file written by `lightning login`.
    lit_user = os.environ.get("LIGHTNING_USER_ID", "")
    lit_key = os.environ.get("LIGHTNING_API_KEY", "")
    lit_source = "environment" if (lit_user and lit_key) else "none"
    if not (lit_user and lit_key):
        cred = Path.home() / ".lightning" / "credentials.json"
        if cred.is_file():
            try:
                import json

                blob = json.loads(cred.read_text(encoding="utf-8"))
                lit_user = lit_user or blob.get("user_id", "")
                lit_key = lit_key or blob.get("api_key", "")
                lit_source = str(cred)
            except (OSError, ValueError):
                lit_source = f"{cred} (unreadable)"

    valid_lightning = (
        bool(lit_user and lit_key)
        and lit_user not in _PLACEHOLDERS
        and lit_key not in _PLACEHOLDERS
    )

    valid_kaggle = bool(user and key) and user not in _PLACEHOLDERS and key not in _PLACEHOLDERS
    valid_hf = bool(hf) and hf not in _PLACEHOLDERS

    return CredentialStatus(
        kaggle=valid_kaggle,
        kaggle_source=kaggle_source if valid_kaggle else "none",
        kaggle_user=user if valid_kaggle else "",
        kaggle_key_fingerprint=fingerprint(key) if valid_kaggle else "-",
        lightning=valid_lightning,
        lightning_source=lit_source if valid_lightning else "none",
        lightning_user=lit_user if valid_lightning else "",
        lightning_key_fingerprint=fingerprint(lit_key) if valid_lightning else "-",
        huggingface=valid_hf,
        hf_source=hf_source if valid_hf else "none",
        hf_token_fingerprint=fingerprint(hf) if valid_hf else "-",
        dotenv_loaded=tuple(applied),
    )


def require(*keys: str, dotenv: str | Path | None = ".env") -> dict[str, str]:
    """Fetch required environment variables, or fail with actionable guidance.

    Args:
        *keys: Variable names that must be set to a non-placeholder value.
        dotenv: ``.env`` path to load first.

    Returns:
        Mapping of key to value, for immediate use by the caller.

    Raises:
        RuntimeError: If any key is missing, naming every missing key and how to
            set it. The message contains no values.
    """
    if dotenv:
        load_dotenv(dotenv)
    found = {k: os.environ.get(k, "") for k in keys}
    missing = [k for k, v in found.items() if not v or v in _PLACEHOLDERS]
    if missing:
        raise RuntimeError(
            f"missing credentials: {', '.join(missing)}\n\n"
            f"Set them in one of:\n"
            f"  1. your shell:   export {missing[0]}=...\n"
            f"  2. a .env file:  cp .env.example .env, then edit it\n"
            f"     (.env is gitignored; never commit or paste it)\n"
            f"  3. for Kaggle:   `kaggle auth login`, or KAGGLE_API_TOKEN, or\n"
            f"                   ~/.kaggle/access_token, or ~/.kaggle/kaggle.json\n\n"
            f"Run `smartscan credentials` to see what is currently configured."
        )
    return found
