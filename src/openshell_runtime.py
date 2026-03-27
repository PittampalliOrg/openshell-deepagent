"""Shared OpenShell runtime helpers used by both Dapr implementations."""

from __future__ import annotations

import json
import os
import threading
from textwrap import dedent
from typing import Any

from openshell import SandboxClient, SandboxSession

SANDBOX_NAME_ENV = "OPENSHELL_SANDBOX_NAME"
DEFAULT_TIMEOUT_SECONDS = 30 * 60


def _merge_output(stdout: str, stderr: str) -> str:
    if stdout and stderr:
        return f"{stdout.rstrip()}\n{stderr.rstrip()}"
    return stdout or stderr


class OpenShellRuntime:
    """Process-local OpenShell session manager with basic sandbox operations."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._session: SandboxSession | None = None
        self._sandbox_name: str | None = None

    def set_sandbox_name(self, name: str) -> None:
        """Set sandbox name for the next session. Resets existing session if name changed."""
        if self._sandbox_name != name:
            with self._lock:
                self._session = None
                self._sandbox_name = name
            os.environ[SANDBOX_NAME_ENV] = name

    def _ensure_session(self) -> SandboxSession:
        if self._session is not None:
            return self._session

        with self._lock:
            if self._session is not None:
                return self._session

            client = SandboxClient.from_active_cluster()
            configured_name = os.getenv(SANDBOX_NAME_ENV)
            if configured_name:
                try:
                    ref = client.get(configured_name)
                except Exception:
                    # Sandbox doesn't exist yet — create a new one
                    import logging
                    _log = logging.getLogger(__name__)
                    _log.info(
                        "Sandbox %r not found, creating a new one", configured_name
                    )
                    ref = client.create()
                    ref = client.wait_ready(ref.name)
                    # Update env var so subsequent calls reuse this sandbox
                    os.environ[SANDBOX_NAME_ENV] = ref.name
                    _log.info(
                        "Created sandbox %r (requested %r)",
                        ref.name, configured_name,
                    )
            else:
                ref = client.create()
                ref = client.wait_ready(ref.name)
                os.environ[SANDBOX_NAME_ENV] = ref.name

            self._sandbox_name = ref.name
            self._session = SandboxSession(client, ref)
            return self._session

    @property
    def sandbox_name(self) -> str:
        self._ensure_session()
        return self._sandbox_name or "unknown"

    def _exec(
        self,
        argv: list[str],
        *,
        stdin: bytes | None = None,
        timeout_seconds: int | None = None,
    ) -> dict[str, Any]:
        session = self._ensure_session()
        result = session.exec(
            argv,
            stdin=stdin,
            timeout_seconds=timeout_seconds or DEFAULT_TIMEOUT_SECONDS,
        )
        return {
            "ok": result.exit_code == 0,
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "output": _merge_output(result.stdout, result.stderr),
            "sandbox_name": self.sandbox_name,
        }

    def execute(self, command: str, timeout_seconds: int | None = None) -> dict[str, Any]:
        """Run an arbitrary shell command inside the sandbox."""
        return self._exec(
            ["bash", "-lc", command],
            timeout_seconds=timeout_seconds,
        )

    def _run_python(
        self,
        script: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        stdin = None
        if payload is not None:
            stdin = json.dumps(payload).encode("utf-8")
        return self._exec(["python3", "-c", script], stdin=stdin)

    def read_file(self, path: str) -> dict[str, Any]:
        """Read a UTF-8 text file from the sandbox."""
        script = dedent(
            """
            import json
            import pathlib
            import sys

            payload = json.loads(sys.stdin.read())
            path = pathlib.Path(payload["path"])
            if not path.exists():
                print(json.dumps({"ok": False, "error": "file_not_found", "path": str(path)}))
                raise SystemExit(0)
            if not path.is_file():
                print(json.dumps({"ok": False, "error": "not_a_file", "path": str(path)}))
                raise SystemExit(0)

            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                print(json.dumps({"ok": False, "error": "binary_file", "path": str(path)}))
                raise SystemExit(0)

            print(json.dumps({"ok": True, "path": str(path), "content": content}))
            """
        ).strip()
        raw = self._run_python(script, {"path": path})
        if not raw["ok"]:
            return raw
        return json.loads(raw["stdout"])

    def write_file(self, path: str, content: str, append: bool = False) -> dict[str, Any]:
        """Write a UTF-8 text file inside the sandbox."""
        script = dedent(
            """
            import json
            import pathlib
            import sys

            payload = json.loads(sys.stdin.read())
            path = pathlib.Path(payload["path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if payload["append"] else "w"
            with path.open(mode, encoding="utf-8") as handle:
                handle.write(payload["content"])
            print(json.dumps({
                "ok": True,
                "path": str(path),
                "append": payload["append"],
                "bytes_written": len(payload["content"].encode("utf-8")),
            }))
            """
        ).strip()
        raw = self._run_python(
            script,
            {"path": path, "content": content, "append": append},
        )
        if not raw["ok"]:
            return raw
        return json.loads(raw["stdout"])

    def edit_file(
        self,
        path: str,
        old_text: str,
        new_text: str,
        replace_all: bool = False,
    ) -> dict[str, Any]:
        """Replace text inside a UTF-8 file in the sandbox."""
        script = dedent(
            """
            import json
            import pathlib
            import sys

            payload = json.loads(sys.stdin.read())
            path = pathlib.Path(payload["path"])
            if not path.exists():
                print(json.dumps({"ok": False, "error": "file_not_found", "path": str(path)}))
                raise SystemExit(0)

            content = path.read_text(encoding="utf-8")
            old_text = payload["old_text"]
            if old_text not in content:
                print(json.dumps({"ok": False, "error": "old_text_not_found", "path": str(path)}))
                raise SystemExit(0)

            count = content.count(old_text)
            if payload["replace_all"]:
                updated = content.replace(old_text, payload["new_text"])
                replacements = count
            else:
                updated = content.replace(old_text, payload["new_text"], 1)
                replacements = 1

            path.write_text(updated, encoding="utf-8")
            print(json.dumps({
                "ok": True,
                "path": str(path),
                "replacements": replacements,
                "replace_all": payload["replace_all"],
            }))
            """
        ).strip()
        raw = self._run_python(
            script,
            {
                "path": path,
                "old_text": old_text,
                "new_text": new_text,
                "replace_all": replace_all,
            },
        )
        if not raw["ok"]:
            return raw
        return json.loads(raw["stdout"])

    def list_files(
        self,
        path: str = ".",
        recursive: bool = False,
        max_entries: int = 200,
    ) -> dict[str, Any]:
        """List files or directory entries inside the sandbox."""
        script = dedent(
            """
            import json
            import pathlib
            import sys

            payload = json.loads(sys.stdin.read())
            path = pathlib.Path(payload["path"])
            if not path.exists():
                print(json.dumps({"ok": False, "error": "path_not_found", "path": str(path)}))
                raise SystemExit(0)

            entries = []
            iterator = path.rglob("*") if payload["recursive"] else path.iterdir()
            for item in iterator:
                entries.append({
                    "path": str(item),
                    "type": "dir" if item.is_dir() else "file",
                })
                if len(entries) >= payload["max_entries"]:
                    break

            print(json.dumps({
                "ok": True,
                "path": str(path),
                "recursive": payload["recursive"],
                "entries": entries,
            }))
            """
        ).strip()
        raw = self._run_python(
            script,
            {
                "path": path,
                "recursive": recursive,
                "max_entries": max_entries,
            },
        )
        if not raw["ok"]:
            return raw
        return json.loads(raw["stdout"])

    def glob_files(self, pattern: str, root: str = ".", max_entries: int = 200) -> dict[str, Any]:
        """Find files matching a glob pattern inside the sandbox."""
        script = dedent(
            """
            import glob
            import json
            import os
            import pathlib
            import sys

            payload = json.loads(sys.stdin.read())
            root = pathlib.Path(payload["root"])
            pattern = os.path.join(str(root), payload["pattern"])
            matches = sorted(glob.glob(pattern, recursive=True))[: payload["max_entries"]]
            print(json.dumps({
                "ok": True,
                "root": str(root),
                "pattern": payload["pattern"],
                "matches": matches,
            }))
            """
        ).strip()
        raw = self._run_python(
            script,
            {"pattern": pattern, "root": root, "max_entries": max_entries},
        )
        if not raw["ok"]:
            return raw
        return json.loads(raw["stdout"])

    def grep_files(
        self,
        pattern: str,
        root: str = ".",
        glob_pattern: str = "*",
        max_matches: int = 200,
    ) -> dict[str, Any]:
        """Search UTF-8 files inside the sandbox using a regular expression."""
        script = dedent(
            """
            import fnmatch
            import json
            import pathlib
            import re
            import sys

            payload = json.loads(sys.stdin.read())
            root = pathlib.Path(payload["root"])
            regex = re.compile(payload["pattern"])
            matches = []

            for file_path in root.rglob("*"):
                if not file_path.is_file():
                    continue
                if not fnmatch.fnmatch(file_path.name, payload["glob_pattern"]):
                    continue
                try:
                    lines = file_path.read_text(encoding="utf-8").splitlines()
                except (UnicodeDecodeError, OSError):
                    continue

                for line_number, line in enumerate(lines, start=1):
                    if regex.search(line):
                        matches.append({
                            "path": str(file_path),
                            "line_number": line_number,
                            "line": line,
                        })
                        if len(matches) >= payload["max_matches"]:
                            print(json.dumps({
                                "ok": True,
                                "root": str(root),
                                "pattern": payload["pattern"],
                                "matches": matches,
                            }))
                            raise SystemExit(0)

            print(json.dumps({
                "ok": True,
                "root": str(root),
                "pattern": payload["pattern"],
                "matches": matches,
            }))
            """
        ).strip()
        raw = self._run_python(
            script,
            {
                "pattern": pattern,
                "root": root,
                "glob_pattern": glob_pattern,
                "max_matches": max_matches,
            },
        )
        if not raw["ok"]:
            return raw
        return json.loads(raw["stdout"])


_RUNTIME = OpenShellRuntime()


def get_runtime() -> OpenShellRuntime:
    """Return the process-wide OpenShell runtime."""
    return _RUNTIME
