#!/usr/bin/env python
"""Drive a Deepnote notebook's kernel from here: run code, launch jobs, poll.

Deepnote's Public API v2 can execute arbitrary code in an interactive session,
which is enough to run training on their GPU without pasting cells by hand.
Two constraints shape everything below.

``POST /sessions/{id}/execute`` caps ``timeoutMs`` at 120000 -- two minutes.
Training takes hours, so anything long is launched **detached** with ``nohup``
into a log file, and progress is read by later two-second calls that tail it.
``launch`` and ``tail`` do exactly that.

``POST /sessions`` *runs the notebook it is anchored to* on creation. This
project is a single-notebook project, so the notebook in question is the user's
own, whose first cell is a 300-episode predictor training run. ``session``
therefore interrupts immediately after creating the session, and says so.

Usage::

    python scripts/deepnote_ctl.py session            # create, print session id
    python scripts/deepnote_ctl.py exec  --code 'print(1)'
    python scripts/deepnote_ctl.py exec  --file local_script.py
    python scripts/deepnote_ctl.py launch --name train --code 'bash commands'
    python scripts/deepnote_ctl.py tail  --name train --lines 40
    python scripts/deepnote_ctl.py status
    python scripts/deepnote_ctl.py stop

The session id is cached in ``.deepnote_session`` so later calls find it.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Remote logs carry progress bars and box-drawing characters. A Windows
# console defaults to cp1252 and raises UnicodeEncodeError on them, which kills
# the poller while the remote job is perfectly healthy -- so never let the
# terminal's encoding decide whether we can read a log.
for _stream in (sys.stdout, sys.stderr):
    # Not a real tty, or already wrapped -- neither is worth failing over.
    with contextlib.suppress(AttributeError, ValueError):
        _stream.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parents[1]
API = "https://api.deepnote.com/v2"
STATE = REPO_ROOT / ".deepnote_session"

#: Single-notebook project SIH26055; the notebook the session anchors to.
NOTEBOOK_ID = os.environ.get("DEEPNOTE_NOTEBOOK_ID", "53af928e0e7547a5b15ceb070f76fe28")

#: Where detached jobs write their logs on the Deepnote machine.
LOG_DIR = "/work/_jobs"


def _key() -> str:
    """API key from the environment, falling back to .env."""
    key = os.environ.get("DEEPNOTE_API_KEY")
    if key:
        return key
    env = REPO_ROOT / ".env"
    if env.is_file():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("DEEPNOTE_API_KEY=") and "=" in line:
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("DEEPNOTE_API_KEY not set and not found in .env")


def call(path: str, method: str = "GET", body: dict | None = None, timeout: int = 180):
    """One API call. Returns ``(status, parsed_or_text)``."""
    req = urllib.request.Request(
        f"{API}{path}", method=method,
        headers={"Authorization": f"Bearer {_key()}", "Content-Type": "application/json"},
        data=json.dumps(body).encode() if body else None,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode()
            try:
                return r.status, json.loads(raw)
            except json.JSONDecodeError:
                return r.status, raw
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:600]


def _session_id(required: bool = True) -> str:
    if STATE.is_file():
        sid = STATE.read_text(encoding="utf-8").strip()
        if sid:
            return sid
    if required:
        raise SystemExit("No session. Run: python scripts/deepnote_ctl.py session")
    return ""


def cmd_session(_args) -> int:
    """Create an interactive session, then interrupt its automatic first run."""
    status, data = call("/sessions", "POST",
                        {"notebookId": NOTEBOOK_ID, "storageMode": "read_write"})
    if status not in (200, 201, 202):
        print(f"create session -> {status}: {data}", file=sys.stderr)
        return 1
    sid = data["session"]["id"] if isinstance(data.get("session"), dict) else data["session"]
    STATE.write_text(sid, encoding="utf-8")
    print(f"session {sid}")

    # Creating a session runs the anchored notebook. That notebook is the
    # user's, and its first cell is a long training run, so stop it at once.
    time.sleep(2)
    st, _ = call(f"/sessions/{sid}/interrupt", "POST", {"notebookId": NOTEBOOK_ID})
    print(f"interrupted the automatic first run -> {st}")
    return 0


def _execute(code: str, timeout_ms: int = 120_000) -> tuple[int, str]:
    """Run code in the kernel; return (status, flattened text output)."""
    sid = _session_id()
    status, data = call(f"/sessions/{sid}/execute", "POST",
                        {"notebookId": NOTEBOOK_ID, "code": code, "timeoutMs": timeout_ms},
                        timeout=timeout_ms // 1000 + 60)
    if status != 200:
        return status, json.dumps(data) if not isinstance(data, str) else data

    chunks: list[str] = []
    for out in (data.get("outputs") or []):
        if isinstance(out, str):
            chunks.append(out)
            continue
        for field in ("text", "content", "value", "traceback", "evalue", "ename"):
            v = out.get(field)
            if isinstance(v, str):
                chunks.append(v)
            elif isinstance(v, list):
                chunks.append("".join(str(x) for x in v))
        data_field = out.get("data")
        if isinstance(data_field, dict):
            for mime in ("text/plain", "text/html"):
                if mime in data_field:
                    v = data_field[mime]
                    chunks.append("".join(v) if isinstance(v, list) else str(v))
    return 200, "\n".join(c for c in chunks if c).strip()


def cmd_exec(args) -> int:
    code = Path(args.file).read_text(encoding="utf-8") if args.file else args.code
    if not code:
        raise SystemExit("pass --code or --file")
    status, text = _execute(code, args.timeout_ms)
    print(text if text else f"(no output)  status={status}")
    return 0 if status == 200 else 1


def cmd_launch(args) -> int:
    """Start a long job detached, so it outlives the two-minute execute cap."""
    code = Path(args.file).read_text(encoding="utf-8") if args.file else args.code
    if not code:
        raise SystemExit("pass --code or --file")
    name = args.name
    script = f"{LOG_DIR}/{name}.sh"
    log = f"{LOG_DIR}/{name}.log"
    payload = (
        "import os, subprocess, textwrap\n"
        f"os.makedirs({LOG_DIR!r}, exist_ok=True)\n"
        f"open({script!r}, 'w').write(textwrap.dedent({code!r}))\n"
        f"subprocess.run(['chmod', '+x', {script!r}])\n"
        # setsid detaches it from the kernel, so an execute timeout or a kernel
        # restart does not take the training with it.
        f"p = subprocess.Popen(['setsid', 'bash', {script!r}],\n"
        f"    stdout=open({log!r}, 'ab'), stderr=subprocess.STDOUT,\n"
        "    start_new_session=True)\n"
        f"open({LOG_DIR + '/' + name + '.pid'!r}, 'w').write(str(p.pid))\n"
        f"print('launched', {name!r}, 'pid', p.pid, '-> {log}')\n"
    )
    status, text = _execute(payload, 60_000)
    print(text or f"status={status}")
    return 0 if status == 200 else 1


def cmd_tail(args) -> int:
    log = f"{LOG_DIR}/{args.name}.log"
    pid = f"{LOG_DIR}/{args.name}.pid"
    code = (
        "import os, subprocess\n"
        f"log, pidf = {log!r}, {pid!r}\n"
        "alive = False\n"
        "if os.path.exists(pidf):\n"
        "    try:\n"
        "        os.kill(int(open(pidf).read().strip()), 0); alive = True\n"
        "    except Exception: alive = False\n"
        "print('RUNNING' if alive else 'not running')\n"
        "if os.path.exists(log):\n"
        f"    print(subprocess.run(['tail','-n','{args.lines}',log],"
        "          capture_output=True, text=True).stdout)\n"
        "else:\n"
        "    print('(no log yet)')\n"
    )
    status, text = _execute(code, 60_000)
    print(text or f"status={status}")
    return 0 if status == 200 else 1


def cmd_status(_args) -> int:
    sid = _session_id(required=False)
    if not sid:
        print("no cached session")
        return 0
    status, data = call(f"/sessions/{sid}/status?notebookId={NOTEBOOK_ID}")
    print(f"session {sid} -> {status}")
    print(json.dumps(data, indent=1)[:800] if not isinstance(data, str) else data)
    return 0


def cmd_stop(_args) -> int:
    sid = _session_id(required=False)
    if not sid:
        print("no cached session")
        return 0
    status, data = call(f"/sessions/{sid}", "DELETE")
    print(f"stop -> {status} {data if isinstance(data, str) else ''}")
    STATE.unlink(missing_ok=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("session").set_defaults(fn=cmd_session)

    e = sub.add_parser("exec")
    e.set_defaults(fn=cmd_exec)
    e.add_argument("--code")
    e.add_argument("--file")
    e.add_argument("--timeout-ms", type=int, default=120_000)

    ln = sub.add_parser("launch")
    ln.set_defaults(fn=cmd_launch)
    ln.add_argument("--name", required=True)
    ln.add_argument("--code")
    ln.add_argument("--file")

    t = sub.add_parser("tail")
    t.set_defaults(fn=cmd_tail)
    t.add_argument("--name", required=True)
    t.add_argument("--lines", type=int, default=40)

    sub.add_parser("status").set_defaults(fn=cmd_status)
    sub.add_parser("stop").set_defaults(fn=cmd_stop)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
