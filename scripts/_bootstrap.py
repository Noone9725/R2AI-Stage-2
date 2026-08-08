"""Bootstrap chung cho moi script CLI.

Cac script nam ngoai package `src`, nen phai chen project root vao sys.path
truoc khi import. Gom o day de khong lap 5 lan.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_settings  # noqa: E402
from src.utils.logging import setup_logging  # noqa: E402


def bootstrap(ensure_dirs: bool = True):
    """Nap settings, bat logging, tao san thu muc du lieu. Tra ve Settings."""
    settings = get_settings()
    log_cfg = settings.logging
    setup_logging(
        level=log_cfg.get("level", "INFO"),
        log_file=settings.root / log_cfg.get("file", "logs/app.log"),
    )
    if ensure_dirs:
        settings.paths.ensure()
    return settings
