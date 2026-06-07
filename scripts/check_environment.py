from __future__ import annotations

import importlib.util
import json
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str
    required: bool = True


def run_command(*args: str) -> tuple[bool, str]:
    executable = shutil.which(args[0])
    if executable is None:
        return False, "not found"

    try:
        result = subprocess.run(
            [executable, *args[1:]],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:  # pragma: no cover - defensive host check
        return False, f"error: {exc}"

    output = (result.stdout or result.stderr).strip().splitlines()
    detail = output[0] if output else f"exit={result.returncode}"
    return result.returncode == 0, detail


def python_module_check(import_name: str, display_name: str, required: bool = False) -> Check:
    found = importlib.util.find_spec(import_name) is not None
    detail = "installed" if found else "not installed"
    return Check(display_name, found, detail, required=required)


def command_check(name: str, *args: str, required: bool = True) -> Check:
    ok, detail = run_command(*args)
    return Check(name, ok, detail, required=required)


def node_version_ok(version_output: str) -> bool:
    raw = version_output.strip().lstrip("v")
    parts = raw.split(".")
    if len(parts) < 2 or not parts[0].isdigit() or not parts[1].isdigit():
        return False
    major = int(parts[0])
    minor = int(parts[1])
    return (major == 20 and minor >= 20) or major >= 22


def collect_checks() -> list[Check]:
    system = platform.system()
    machine = platform.machine()
    mem_ok, mem_detail = run_command("sysctl", "-n", "hw.memsize")
    memory_gb = "unknown"
    if mem_ok and mem_detail.isdigit():
        memory_gb = f"{int(mem_detail) / 1024**3:.0f} GB"

    node_ok, node_detail = run_command("node", "--version")
    node_ready = node_ok and node_version_ok(node_detail)

    checks = [
        Check("macOS host", system == "Darwin", f"{system} {platform.release()}"),
        Check("Apple Silicon", machine == "arm64", machine),
        Check("host memory", mem_ok, memory_gb, required=False),
        command_check("uv", "uv", "--version"),
        command_check("git", "git", "--version"),
        Check("Node.js for promptfoo", node_ready, node_detail if node_ok else "not found"),
        command_check("npm", "npm", "--version"),
        python_module_check("mlx_lm", "mlx-lm", required=False),
        python_module_check("argilla", "argilla SDK", required=False),
        python_module_check("deepeval", "DeepEval", required=False),
        command_check("promptfoo", "promptfoo", "--version", required=False),
    ]
    return checks


def print_text(checks: list[Check]) -> None:
    print("Phase 1 environment check")
    print(f"repo: {REPO_ROOT}")
    for check in checks:
        mark = "OK" if check.ok else ("WARN" if not check.required else "FAIL")
        print(f"[{mark}] {check.name}: {check.detail}")


def main() -> int:
    json_output = "--json" in sys.argv
    checks = collect_checks()
    if json_output:
        print(json.dumps([asdict(check) for check in checks], ensure_ascii=False, indent=2))
    else:
        print_text(checks)

    return 1 if any(check.required and not check.ok for check in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
