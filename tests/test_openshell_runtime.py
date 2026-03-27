from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from dataclasses import dataclass

from src.openshell_runtime import OpenShellRuntime


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    exit_code: int


class FakeSession:
    id = "fake-session"

    def exec(self, argv, stdin=None, timeout_seconds=None):  # noqa: ARG002
        cwd = os.getcwd()
        completed = subprocess.run(
            argv,
            input=stdin,
            capture_output=True,
            cwd=cwd,
            check=False,
        )
        return ExecResult(
            stdout=completed.stdout.decode("utf-8"),
            stderr=completed.stderr.decode("utf-8"),
            exit_code=completed.returncode,
        )


class OpenShellRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.old_cwd = os.getcwd()
        os.chdir(self.tempdir.name)
        self.runtime = OpenShellRuntime()
        self.runtime._session = FakeSession()
        self.runtime._sandbox_name = "test-sandbox"

    def tearDown(self) -> None:
        os.chdir(self.old_cwd)
        self.tempdir.cleanup()

    def test_write_read_and_edit_file(self) -> None:
        write_result = self.runtime.write_file("sandbox/hello.txt", "hello world")
        self.assertTrue(write_result["ok"])

        read_result = self.runtime.read_file("sandbox/hello.txt")
        self.assertEqual(read_result["content"], "hello world")

        edit_result = self.runtime.edit_file(
            "sandbox/hello.txt",
            "world",
            "dapr",
        )
        self.assertTrue(edit_result["ok"])

        reread = self.runtime.read_file("sandbox/hello.txt")
        self.assertEqual(reread["content"], "hello dapr")

    def test_list_glob_and_grep(self) -> None:
        self.runtime.write_file("sandbox/a.txt", "alpha\nbeta\n")
        self.runtime.write_file("sandbox/nested/b.txt", "gamma beta\n")

        listed = self.runtime.list_files("sandbox", recursive=True)
        self.assertTrue(listed["entries"])

        globbed = self.runtime.glob_files("**/*.txt", root="sandbox")
        self.assertEqual(len(globbed["matches"]), 2)

        grepped = self.runtime.grep_files("beta", root="sandbox", glob_pattern="*.txt")
        self.assertEqual(len(grepped["matches"]), 2)


if __name__ == "__main__":
    unittest.main()
