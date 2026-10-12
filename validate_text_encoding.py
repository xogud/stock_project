from __future__ import annotations

from pathlib import Path


TEXT_EXTENSIONS = {".md", ".py", ".sql", ".txt", ".json"}
SKIP_DIRS = {".git", "__pycache__", ".cache", "models", "data", "logs"}
SUSPECT_MARKERS = [
    "\ufffd",
    "?꾪",
    "?댁",
    "?섏",
    "?좎",
    "?쒖",
    "嫄곕옒",
    "二쇱",
    "誘몃",
    "理",
]


def candidate_files(root: Path) -> list[Path]:
    files = []
    for path in root.rglob("*"):
        if path.is_dir():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix.lower() in TEXT_EXTENSIONS:
            files.append(path)
    return sorted(files)


def check_file(path: Path) -> list[str]:
    if path.name == Path(__file__).name:
        return []

    problems = []
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        return [f"not valid UTF-8: {exc}"]

    for marker in SUSPECT_MARKERS:
        if marker in text:
            problems.append(f"suspect marker found: {marker}")
    return problems


def main() -> None:
    root = Path(".")
    failures = []
    for path in candidate_files(root):
        problems = check_file(path)
        if problems:
            failures.append((path, problems))

    if failures:
        for path, problems in failures:
            print(path)
            for problem in problems:
                print(f"  - {problem}")
        raise SystemExit(1)

    print("encoding_check=ok")


if __name__ == "__main__":
    main()
