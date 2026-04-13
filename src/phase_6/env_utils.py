from __future__ import annotations

import os
from pathlib import Path


def load_phase6_env(project_root: Path) -> list[Path]:
    """Load local env files for Phase 6, preferring .env.local over .env."""
    loaded: list[Path] = []
    for name in (".env.local", ".env"):
        path = project_root / name
        if path.exists():
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())
            loaded.append(path)
    return loaded
