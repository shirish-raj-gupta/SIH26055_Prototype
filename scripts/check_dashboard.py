#!/usr/bin/env python
"""Drive the Streamlit demo in a real browser and check what a judge must see.

The dashboard is the deliverable a judge actually looks at, and it is the one
part of this repository the test suite cannot reach: ``tests/test_dashboard.py``
imports the module and exercises its pure functions, but nothing renders the
page, so a layout that raises on first paint would pass CI and fail live.

    pip install playwright
    python -m playwright install chromium
    python scripts/check_dashboard.py

Starts the app on a free port, loads it in headless Chromium, asserts the things
the demo has to communicate without a briefing, and writes screenshots to
``reports/dashboard/``. Exits non-zero on the first failed check.

Add ``--headed`` to watch it happen, ``--keep`` to leave the server running.
"""

from __future__ import annotations

import argparse
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SHOTS = REPO_ROOT / "reports" / "dashboard"

#: Text that must be on the page. Each entry is something a judge needs in order
#: to read the waterfall or the gauges without being told what they mean.
REQUIRED_TEXT = (
    "ANVESHAK",
    "The receiver sees 1 slice of the band at a time",
    "transmitted, missed",          # colour key
    "intercepted",
    "where the receiver is looking",
    "pop-up threat appears",
    "Emitters found",
    "Interception ratio",
    "Band coverage",
    "Scheduler reasoning",
)

# In A/B mode each metric column is captioned with its slot letter and agent, so
# the two identical stacks of numbers can be told apart.
REQUIRED_AFTER_RENDER = ("A ·", "B ·")


def _free_port() -> int:
    """Return a port the OS has just confirmed is free."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_for_server(url: str, proc: subprocess.Popen, timeout: float = 90.0) -> None:
    """Block until the app answers, or fail with whatever it printed instead.

    Args:
        url: Base URL to poll.
        proc: The server process, watched so a crash fails fast.
        timeout: Seconds to wait before giving up.

    Raises:
        RuntimeError: If the server exits or never becomes ready.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            out = (proc.stdout.read() if proc.stdout else "") or "(no output)"
            raise RuntimeError(f"streamlit exited with {proc.returncode}:\n{out}")
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(0.5)
    raise RuntimeError(f"streamlit did not answer on {url} within {timeout:.0f}s")


def main() -> int:
    """Run the browser checks."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--headed", action="store_true", help="Show the browser.")
    ap.add_argument("--keep", action="store_true", help="Leave the server up on exit.")
    args = ap.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright is not installed:\n"
              "  pip install playwright\n"
              "  python -m playwright install chromium", file=sys.stderr)
        return 2

    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    SHOTS.mkdir(parents=True, exist_ok=True)

    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "dashboard/app.py",
         "--server.port", str(port), "--server.headless", "true",
         "--browser.gatherUsageStats", "false"],
        cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )

    failures: list[str] = []
    try:
        print(f"starting streamlit on {url} ...", flush=True)
        _wait_for_server(url, proc)
        print("server up", flush=True)

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=not args.headed)
            page = browser.new_page(viewport={"width": 1680, "height": 1000})
            console: list[str] = []
            page.on("console", lambda m: console.append(m.text) if m.type == "error" else None)

            page.goto(url, wait_until="load", timeout=60_000)
            # Streamlit paints a skeleton first; wait for real content.
            page.wait_for_selector("text=ANVESHAK", timeout=60_000)
            page.wait_for_timeout(4_000)   # let the first episode render

            body = page.inner_text("body")
            for needle in REQUIRED_TEXT + REQUIRED_AFTER_RENDER:
                if needle not in body:
                    failures.append(f"missing from page: {needle!r}")

            # A Streamlit exception renders as a dedicated element; the page can
            # otherwise look fine while the panel that matters is a stack trace.
            n_exc = page.locator('[data-testid="stException"]').count()
            if n_exc:
                failures.append(f"{n_exc} Streamlit exception(s) rendered")
                print(page.locator('[data-testid="stException"]').first.inner_text()[:2000],
                      file=sys.stderr)

            # The waterfall is the whole argument; A/B mode must draw two.
            n_plots = page.locator(".js-plotly-plot").count()
            if n_plots < 2:
                failures.append(f"expected 2 waterfalls in A/B mode, found {n_plots}")

            page.screenshot(path=str(SHOTS / "01_initial.png"), full_page=True)

            # Step once and confirm the clock advances rather than sitting at 0.
            # This assertion is the point of the check: the first version only
            # clicked and screenshotted, and so passed while the header clock sat
            # at 0.00 s because it rendered before the tracks were advanced.
            try:
                page.get_by_role("button", name="Step").click(timeout=10_000)
                page.wait_for_timeout(3_000)
                page.screenshot(path=str(SHOTS / "02_after_step.png"), full_page=True)
                stepped = page.inner_text("body")
                m = re.search(r"t\s*=\s*([0-9.]+)\s*s", stepped)
                if m is None:
                    failures.append("no clock reading found on the page after Step")
                elif float(m.group(1)) <= 0.0:
                    failures.append(
                        f"clock still reads t = {m.group(1)} s after Step; the header "
                        "is rendering before the tracks advance"
                    )
                if "stStatusWidget" not in stepped and "Emitters found" not in stepped:
                    failures.append("metric column vanished after Step")
            except Exception as exc:
                failures.append(f"Step button did not work: {exc}")

            for msg in console:
                if "Warning" not in msg:
                    failures.append(f"console error: {msg[:200]}")

            browser.close()
    finally:
        if not args.keep:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    print()
    if failures:
        print(f"FAILED ({len(failures)}):", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print(f"all checks passed; screenshots in {SHOTS.relative_to(REPO_ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
