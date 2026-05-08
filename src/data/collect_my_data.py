from __future__ import annotations

from pathlib import Path


def collect_my_data(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    raise NotImplementedError("Webcam recording flow is project-specific and should be implemented here.")


if __name__ == "__main__":
    raise SystemExit("Import and call collect_my_data() from your own capture workflow.")
