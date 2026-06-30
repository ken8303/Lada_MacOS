#!/usr/bin/env python3
import os
import stat
import sys
from pathlib import Path

UF_DATALESS = 0x40000000


def should_skip(relative_path: str, excludes: set[str]) -> bool:
    if not relative_path:
        return False
    parts = Path(relative_path).parts
    if "__pycache__" in parts:
        return True
    if any(part in excludes for part in parts):
        return True
    for index in range(len(parts)):
        candidate = "/".join(parts[: index + 1])
        if candidate in excludes:
            return True
    return False


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    if os.environ.get("LADA_COPY_TRACE") == "1":
        Path("/private/tmp/lada-copy-tree-current.txt").write_text(
            f"{source}\n{destination}\n",
            encoding="utf-8",
        )
    try:
        source_size = source.stat().st_size
    except OSError:
        source_size = 0
    if source_size > 1024 * 1024:
        try:
            os.link(source, destination)
            return
        except OSError:
            pass

    with source.open("rb") as input_file, destination.open("wb") as output_file:
        while True:
            chunk = input_file.read(1024 * 1024)
            if not chunk:
                break
            output_file.write(chunk)
    try:
        source_mode = stat.S_IMODE(source.lstat().st_mode)
        os.chmod(destination, source_mode, follow_symlinks=False)
    except OSError:
        pass


def copy_tree(source: Path, destination: Path, excludes: set[str]) -> None:
    source = source.resolve()
    destination.mkdir(parents=True, exist_ok=True)

    for root, dirnames, filenames in os.walk(source, topdown=True, followlinks=False):
        root_path = Path(root)
        relative_root = root_path.relative_to(source).as_posix()
        if relative_root == ".":
            relative_root = ""

        dirnames[:] = [
            dirname
            for dirname in dirnames
            if not should_skip(
                f"{relative_root}/{dirname}".strip("/"),
                excludes,
            )
        ]

        destination_root = destination / relative_root
        destination_root.mkdir(parents=True, exist_ok=True)

        for dirname in dirnames:
            source_dir = root_path / dirname
            destination_dir = destination_root / dirname
            if source_dir.is_symlink():
                if destination_dir.exists() or destination_dir.is_symlink():
                    destination_dir.unlink()
                os.symlink(os.readlink(source_dir), destination_dir)
            else:
                destination_dir.mkdir(parents=True, exist_ok=True)

        for filename in filenames:
            relative_file = f"{relative_root}/{filename}".strip("/")
            if should_skip(relative_file, excludes):
                continue

            source_file = root_path / filename
            destination_file = destination_root / filename
            if source_file.is_symlink():
                if destination_file.exists() or destination_file.is_symlink():
                    destination_file.unlink()
                os.symlink(os.readlink(source_file), destination_file)
            else:
                copy_file(source_file, destination_file)


def list_dataless(source: Path, excludes: set[str]) -> int:
    source = source.resolve()
    count = 0
    for root, dirnames, filenames in os.walk(source, topdown=True, followlinks=False):
        root_path = Path(root)
        relative_root = root_path.relative_to(source).as_posix()
        if relative_root == ".":
            relative_root = ""

        dirnames[:] = [
            dirname
            for dirname in dirnames
            if not should_skip(
                f"{relative_root}/{dirname}".strip("/"),
                excludes,
            )
        ]

        for filename in filenames:
            relative_file = f"{relative_root}/{filename}".strip("/")
            if should_skip(relative_file, excludes):
                continue

            source_file = root_path / filename
            try:
                flags = source_file.stat().st_flags
            except (AttributeError, OSError):
                continue
            if flags & UF_DATALESS:
                print(source_file)
                count += 1
    return count


def main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] == "--list-dataless":
        if len(sys.argv) < 3:
            print("usage: copy_tree.py --list-dataless SOURCE [EXCLUDE ...]", file=sys.stderr)
            return 2
        source = Path(sys.argv[2])
        excludes = set(sys.argv[3:])
        list_dataless(source, excludes)
        return 0

    if len(sys.argv) < 3:
        print("usage: copy_tree.py SOURCE DESTINATION [EXCLUDE ...]", file=sys.stderr)
        return 2

    source = Path(sys.argv[1])
    destination = Path(sys.argv[2])
    excludes = set(sys.argv[3:])
    copy_tree(source, destination, excludes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
