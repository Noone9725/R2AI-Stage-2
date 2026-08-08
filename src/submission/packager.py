"""Dong goi bai nop thanh ZIP dung yeu cau BTC.

Yeu cau nghiem ngat, sai la bi tu choi:
  - submission.json va data/ nam NGAY o goc ZIP, khong bao boi folder cha
  - trong ZIP chi co DUNG MOT file .json
  - chi copy nhung CSV thuc su duoc tham chieu (ZIP nhe, tranh file rac)
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from ..config import get_settings
from ..schemas import SubmissionItem
from ..utils.io import read_json, write_json
from ..utils.logging import get_logger
from .validator import ValidationReport, validate_submission

log = get_logger(__name__)


class SubmissionPackager:
    def __init__(self, source_data_dir: Path | None = None, csv_prefix: str | None = None):
        settings = get_settings()
        self.source_data_dir = source_data_dir or settings.paths.processed
        self.csv_prefix = csv_prefix or settings.submission.get("csv_prefix", "data/")
        self.outputs = settings.paths.outputs

    # ── stage ─────────────────────────────────────────────

    def stage(self, items: list[SubmissionItem], stage_dir: Path | str) -> Path:
        """Dung cay thu muc dung hinh dang ZIP, chua zip."""
        stage = Path(stage_dir)
        if stage.exists():
            shutil.rmtree(stage)
        (stage / "data").mkdir(parents=True, exist_ok=True)

        payload = [it.to_dict() for it in items]
        write_json(payload, stage / "submission.json")

        needed = {
            ev["csv_path"]
            for it in payload
            for ev in it["evidence"]
        }
        copied, missing = 0, []
        for rel in sorted(needed):
            src = self.source_data_dir / Path(rel).name
            if not src.exists():
                missing.append(rel)
                continue
            dst = stage / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1

        if missing:
            log.warning("Thieu %d CSV nguon, vd: %s", len(missing), missing[:5])
        log.info("Staged %d cau hoi, %d CSV -> %s", len(items), copied, stage)
        return stage

    # ── validate + zip ────────────────────────────────────

    def validate(
        self, stage_dir: Path | str, expected_ids: list[int] | None = None
    ) -> ValidationReport:
        stage = Path(stage_dir)
        items = read_json(stage / "submission.json")
        return validate_submission(
            items,
            root_dir=stage,
            expected_ids=expected_ids,
            csv_prefix=self.csv_prefix,
        )

    def zip(self, stage_dir: Path | str, zip_path: Path | str | None = None) -> Path:
        stage = Path(stage_dir)
        target = Path(zip_path) if zip_path else self.outputs / "submissions" / f"{stage.name}.zip"
        target.parent.mkdir(parents=True, exist_ok=True)

        jsons = list(stage.rglob("*.json"))
        if len(jsons) != 1:
            raise ValueError(f"ZIP phai chua dung 1 file .json, dang co {len(jsons)}")

        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    zf.write(path, path.relative_to(stage).as_posix())

        size_mb = target.stat().st_size / 1024 / 1024
        log.info("Da tao %s (%.1f MB)", target, size_mb)
        return target

    def package(
        self,
        items: list[SubmissionItem],
        name: str = "submission",
        expected_ids: list[int] | None = None,
        strict: bool = True,
    ) -> tuple[Path, ValidationReport]:
        """stage -> validate -> zip. strict=True thi loi la dung, khong zip."""
        stage = self.stage(items, self.outputs / name)
        report = self.validate(stage, expected_ids=expected_ids)

        print(report.render())
        if not report.ok and strict:
            raise ValueError(
                f"Bai nop co {len(report.errors)} loi — khong zip. "
                "Dat strict=False de zip du sao."
            )

        return self.zip(stage, self.outputs / "submissions" / f"{name}.zip"), report
