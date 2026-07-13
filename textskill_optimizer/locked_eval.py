"""Seal a test split and allow exactly one final evaluation attempt."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import stat
import subprocess
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from cryptography.fernet import Fernet, InvalidToken


LOCK_FORMAT = "textskill-fernet-tar-gz-v1"


def seal_directory(
    source_dir: str | Path,
    archive_path: str | Path,
    key_path: str | Path,
    lock_path: str | Path,
    *,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source = Path(source_dir).resolve()
    archive = Path(archive_path)
    key_file = Path(key_path)
    lock_file = Path(lock_path)
    if not source.is_dir():
        raise ValueError(f"source_dir must be a directory: {source}")
    for target in (archive, key_file, lock_file):
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite locked-eval artifact: {target}")

    plaintext = build_tar_gz(source)
    key = Fernet.generate_key()
    ciphertext = Fernet(key).encrypt(plaintext)
    digest = hashlib.sha256(ciphertext).hexdigest()

    archive.parent.mkdir(parents=True, exist_ok=True)
    key_file.parent.mkdir(parents=True, exist_ok=True)
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    archive.write_bytes(ciphertext)
    key_file.write_bytes(key + b"\n")
    key_file.chmod(stat.S_IRUSR | stat.S_IWUSR)

    payload = {
        "format": LOCK_FORMAT,
        "sealed_at": utc_now(),
        "archive": str(archive),
        "archive_sha256": digest,
        "archive_bytes": len(ciphertext),
        "details": details or {},
    }
    lock_file.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def build_tar_gz(source: Path) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in sorted(source.rglob("*")):
            archive.add(path, arcname=str(path.relative_to(source)), recursive=False)
    return buffer.getvalue()


def run_locked_archive(
    archive_path: str | Path,
    key_path: str | Path,
    lock_path: str | Path,
    receipt_path: str | Path,
    command: Sequence[str],
    *,
    cwd: str | Path | None = None,
    extra_env: dict[str, str] | None = None,
) -> int:
    archive = Path(archive_path)
    key_file = Path(key_path)
    lock_file = Path(lock_path)
    receipt = Path(receipt_path)
    if receipt.exists():
        raise RuntimeError(f"Final evaluation already consumed; receipt exists: {receipt}")
    if not command:
        raise ValueError("command must not be empty")

    lock = json.loads(lock_file.read_text(encoding="utf-8"))
    verify_lock(archive, lock)
    try:
        plaintext = Fernet(key_file.read_bytes().strip()).decrypt(archive.read_bytes())
    except (ValueError, InvalidToken) as exc:
        raise ValueError("Locked test key is invalid") from exc

    with tempfile.TemporaryDirectory(prefix="textskill-locked-test-") as tmp:
        root = Path(tmp) / "benchmark"
        root.mkdir()
        extract_tar_gz_safely(plaintext, root)
        task_file = root / str(lock.get("details", {}).get("task_file", "test.jsonl"))
        if not task_file.is_file():
            raise ValueError(f"Locked test task file is missing: {task_file}")

        env = os.environ.copy()
        env.update(extra_env or {})
        env["SKILLOPT_LOCKED_TEST_ROOT"] = str(root)
        env["CROSS_AGENT_TASKS"] = str(task_file)
        started_at = utc_now()
        returncode: int | None = None
        error: str | None = None
        try:
            completed = subprocess.run(
                list(command),
                cwd=Path(cwd).resolve() if cwd is not None else None,
                env=env,
                check=False,
            )
            returncode = completed.returncode
            return returncode
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            receipt.parent.mkdir(parents=True, exist_ok=True)
            receipt.write_text(
                json.dumps(
                    {
                        "format": LOCK_FORMAT,
                        "archive_sha256": lock["archive_sha256"],
                        "started_at": started_at,
                        "finished_at": utc_now(),
                        "command": list(command),
                        "returncode": returncode,
                        "error": error,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )


def verify_lock(archive: Path, lock: dict[str, Any]) -> None:
    if lock.get("format") != LOCK_FORMAT:
        raise ValueError(f"Unsupported lock format: {lock.get('format')!r}")
    actual = hashlib.sha256(archive.read_bytes()).hexdigest()
    expected = str(lock.get("archive_sha256", ""))
    if actual != expected:
        raise ValueError("Locked test archive hash does not match its commitment")


def extract_tar_gz_safely(payload: bytes, destination: Path) -> None:
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        destination_root = destination.resolve()
        for member in archive.getmembers():
            if member.issym() or member.islnk():
                raise ValueError(f"Locked test archive contains a link: {member.name}")
            target = (destination / member.name).resolve()
            if target != destination_root and destination_root not in target.parents:
                raise ValueError(f"Locked test archive escapes destination: {member.name}")
        archive.extractall(destination)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)

    seal = subparsers.add_parser("seal")
    seal.add_argument("--source", required=True)
    seal.add_argument("--archive", required=True)
    seal.add_argument("--key-file", required=True)
    seal.add_argument("--lock", required=True)
    seal.add_argument("--details-json")

    run = subparsers.add_parser("run")
    run.add_argument("--archive", required=True)
    run.add_argument("--key-file", required=True)
    run.add_argument("--lock", required=True)
    run.add_argument("--receipt", required=True)
    run.add_argument("--cwd")
    run.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.action == "seal":
        details = (
            json.loads(Path(args.details_json).read_text(encoding="utf-8"))
            if args.details_json
            else {}
        )
        payload = seal_directory(
            args.source,
            args.archive,
            args.key_file,
            args.lock,
            details=details,
        )
        print(json.dumps(payload, sort_keys=True))
        return 0

    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    return run_locked_archive(
        args.archive,
        args.key_file,
        args.lock,
        args.receipt,
        command,
        cwd=args.cwd,
    )


if __name__ == "__main__":
    raise SystemExit(main())
