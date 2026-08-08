"""Test tinh toan ven artifact corpus/index — Phase 3.

Bug goc: mot lan chay `--limit 20` da ghi de manifest cua 125.839 CSV,
chi con 1.092 dong. CSV van con tren dia nhung mat hoan toan reference,
va khong co exception/canh bao nao.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.normalization.csv_writer import CsvWriter
from src.pipeline.integrity import check_corpus, check_index
from src.schemas import ExtractedTable
from src.utils.io import read_jsonl, write_jsonl_atomic


# ── helpers ───────────────────────────────────────────────


def _table(doc_id: str, position: int) -> ExtractedTable:
    return ExtractedTable(doc_id=doc_id, position=position, title=f"Bang {position}")


def _df() -> pd.DataFrame:
    return pd.DataFrame({"item": ["Tien"], "year": [2020], "value": [1.0]})


def _writer(tmp_path: Path) -> CsvWriter:
    return CsvWriter(processed_dir=tmp_path / "processed")


def _write_n(writer: CsvWriter, doc: str, positions: range) -> None:
    for pos in positions:
        writer.write(_table(doc, pos), _df(), card=f"card {pos}", ticker="AAA", year=2020)


# ── Test A — full build ───────────────────────────────────


def test_A_full_build_ghi_dung_so_entry(tmp_path: Path) -> None:
    w = _writer(tmp_path)
    _write_n(w, "DOC_2020", range(1, 11))
    target = tmp_path / "manifest.jsonl"

    w.write_manifest(target, mode="full")

    assert len(list(read_jsonl(target))) == 10


# ── Test B — limited build KHONG pha manifest ─────────────


def test_B_upsert_khong_truncate_manifest_cu(tmp_path: Path) -> None:
    """Day chinh la bug da xay ra: 100 entry -> chay limited 20 -> con 20."""
    target = tmp_path / "manifest.jsonl"

    w1 = _writer(tmp_path)
    _write_n(w1, "DOC_FULL", range(1, 101))
    w1.write_manifest(target, mode="full")
    assert len(list(read_jsonl(target))) == 100

    # Lan chay giới han, cham 20 bang KHAC
    w2 = _writer(tmp_path)
    _write_n(w2, "DOC_LIMITED", range(1, 21))
    w2.write_manifest(target, mode="upsert")

    rows = list(read_jsonl(target))
    assert len(rows) == 120, "upsert phai giu 100 dong cu + them 20 dong moi"
    assert len({r["table_ref"] for r in rows}) == 120


def test_B2_full_mode_van_thay_the_co_chu_dich(tmp_path: Path) -> None:
    """mode='full' van phai thay the — do la y dinh cua full rebuild."""
    target = tmp_path / "manifest.jsonl"
    w1 = _writer(tmp_path)
    _write_n(w1, "DOC_A", range(1, 51))
    w1.write_manifest(target, mode="full")

    w2 = _writer(tmp_path)
    _write_n(w2, "DOC_B", range(1, 6))
    w2.write_manifest(target, mode="full")

    assert len(list(read_jsonl(target))) == 5


def test_B3_mode_khong_hop_le_bi_tu_choi(tmp_path: Path) -> None:
    w = _writer(tmp_path)
    _write_n(w, "DOC", range(1, 3))
    with pytest.raises(ValueError, match="mode"):
        w.write_manifest(tmp_path / "m.jsonl", mode="append")


# ── Test C — upsert khong tao duplicate ───────────────────


def test_C_upsert_cung_bang_khong_sinh_trung(tmp_path: Path) -> None:
    target = tmp_path / "manifest.jsonl"

    w1 = _writer(tmp_path)
    _write_n(w1, "DOC_X", range(1, 11))
    w1.write_manifest(target, mode="full")

    # Xu ly LAI dung nhung bang do
    w2 = _writer(tmp_path)
    _write_n(w2, "DOC_X", range(1, 11))
    w2.write_manifest(target, mode="upsert")

    rows = list(read_jsonl(target))
    assert len(rows) == 10, "reprocess cung bang khong duoc nhan doi entry"
    refs = [r["table_ref"] for r in rows]
    assert len(refs) == len(set(refs))


def test_C2_upsert_ghi_de_ban_moi_nhat(tmp_path: Path) -> None:
    target = tmp_path / "manifest.jsonl"
    w1 = _writer(tmp_path)
    w1.write(_table("DOC_Y", 1), _df(), card="CU", ticker="AAA", year=2020)
    w1.write_manifest(target, mode="full")

    w2 = _writer(tmp_path)
    w2.write(_table("DOC_Y", 1), _df(), card="MOI", ticker="AAA", year=2020)
    w2.write_manifest(target, mode="upsert")

    rows = list(read_jsonl(target))
    assert len(rows) == 1
    assert rows[0]["card"] == "MOI"


# ── Test D — manifest tro toi CSV khong ton tai ───────────


def test_D_missing_csv_bi_bat(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl_atomic(
        [{"table_ref": "DOC|1", "doc_id": "DOC", "position": 1,
          "filename": "khong_ton_tai.csv", "ticker": "AAA"}],
        manifest,
    )

    rep = check_corpus(manifest_path=manifest, processed_dir=processed,
                       raw_dir=tmp_path / "raw")
    assert rep.missing_csv == 1
    assert not rep.ok
    assert any("CSV khong ton tai" in e for e in rep.errors)


# ── Test E — orphan CSV (bug thuc te) ─────────────────────


def test_E_orphan_csv_bi_phat_hien(tmp_path: Path) -> None:
    """125.839 CSV tren dia, manifest 1.092 dong -> phai la LOI."""
    processed = tmp_path / "processed"
    processed.mkdir()
    for i in range(100):
        (processed / f"DOC_table_{i}.csv").write_text("item,value\na,1\n", encoding="utf-8")

    manifest = tmp_path / "manifest.jsonl"
    write_jsonl_atomic(
        [{"table_ref": "DOC|0", "doc_id": "DOC", "position": 0,
          "filename": "DOC_table_0.csv", "ticker": "AAA"}],
        manifest,
    )

    rep = check_corpus(manifest_path=manifest, processed_dir=processed,
                       raw_dir=tmp_path / "raw")
    assert rep.orphan_csv == 99
    assert not rep.ok, "99% CSV mo coi phai la LOI, khong phai canh bao"
    assert any("mo coi" in e or "KHONG co trong manifest" in e for e in rep.errors)


def test_E2_vai_orphan_chi_la_canh_bao(tmp_path: Path) -> None:
    """Vai file mo coi (doi tham so extraction) la binh thuong."""
    processed = tmp_path / "processed"
    processed.mkdir()
    rows = []
    for i in range(100):
        (processed / f"DOC_table_{i}.csv").write_text("item,value\na,1\n", encoding="utf-8")
        if i < 98:
            rows.append({"table_ref": f"DOC|{i}", "doc_id": "DOC", "position": i,
                         "filename": f"DOC_table_{i}.csv", "ticker": "AAA"})
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl_atomic(rows, manifest)

    rep = check_corpus(manifest_path=manifest, processed_dir=processed,
                       raw_dir=tmp_path / "raw")
    assert rep.orphan_csv == 2
    assert rep.ok, "2% mo coi -> canh bao, khong chan pipeline"
    assert rep.warnings


def test_E3_duplicate_table_ref_bi_bat(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / "DOC_table_1.csv").write_text("item,value\na,1\n", encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl_atomic(
        [{"table_ref": "DOC|1", "doc_id": "DOC", "position": 1,
          "filename": "DOC_table_1.csv", "ticker": "AAA"}] * 2,
        manifest,
    )
    rep = check_corpus(manifest_path=manifest, processed_dir=processed,
                       raw_dir=tmp_path / "raw")
    assert rep.duplicate_table_ids == 1
    assert not rep.ok


# ── Test F — index/manifest mismatch ──────────────────────


def _corpus_ok(tmp_path: Path, n: int = 10):
    processed = tmp_path / "processed"
    processed.mkdir(exist_ok=True)
    rows = []
    for i in range(n):
        (processed / f"DOC_table_{i}.csv").write_text("item,value\na,1\n", encoding="utf-8")
        rows.append({"table_ref": f"DOC|{i}", "doc_id": "DOC", "position": i,
                     "filename": f"DOC_table_{i}.csv", "ticker": "AAA"})
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl_atomic(rows, manifest)
    return check_corpus(manifest_path=manifest, processed_dir=processed,
                        raw_dir=tmp_path / "raw")


def test_F_index_count_lech_manifest_thi_fail(tmp_path: Path) -> None:
    from src.utils.io import save_pickle

    rep = _corpus_ok(tmp_path, n=10)
    assert rep.ok

    index_dir = tmp_path / "index"
    index_dir.mkdir()
    save_pickle({"ids": [f"DOC|{i}" for i in range(4)]}, index_dir / "bm25.pkl")
    save_pickle({"ids": [f"DOC|{i}" for i in range(10)]}, index_dir / "vectors.pkl")

    out = check_index(rep, index_dir=index_dir, require_dense=True)
    assert out.bm25_entries == 4
    assert not out.ok
    assert any("BM25" in e and "manifest" in e for e in out.errors)


def test_F2_thieu_dense_index_bi_bat(tmp_path: Path) -> None:
    """Truong hop thuc te: vectors.pkl khong ton tai."""
    from src.utils.io import save_pickle

    rep = _corpus_ok(tmp_path, n=5)
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    save_pickle({"ids": [f"DOC|{i}" for i in range(5)]}, index_dir / "bm25.pkl")

    out = check_index(rep, index_dir=index_dir, require_dense=True)
    assert not out.ok
    assert any("dense index" in e for e in out.errors)


def test_F3_index_khop_thi_ok(tmp_path: Path) -> None:
    from src.utils.io import save_pickle

    rep = _corpus_ok(tmp_path, n=7)
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    ids = [f"DOC|{i}" for i in range(7)]
    save_pickle({"ids": ids}, index_dir / "bm25.pkl")
    save_pickle({"ids": ids}, index_dir / "vectors.pkl")

    out = check_index(rep, index_dir=index_dir, require_dense=True)
    assert out.ok, out.render()
    assert out.bm25_entries == out.dense_entries == 7


# ── Test G — atomic write ─────────────────────────────────


def test_G_ghi_that_bai_giu_nguyen_file_cu(tmp_path: Path) -> None:
    """Serialize loi giua chung -> manifest cu phai con nguyen ven."""
    target = tmp_path / "manifest.jsonl"
    write_jsonl_atomic([{"table_ref": "DOC|1", "keep": "me"}], target)
    before = target.read_text(encoding="utf-8")

    def bad_rows():
        yield {"table_ref": "DOC|2"}
        raise RuntimeError("het dia")

    with pytest.raises(RuntimeError, match="het dia"):
        write_jsonl_atomic(bad_rows(), target)

    assert target.read_text(encoding="utf-8") == before, "file cu bi hong"
    assert json.loads(before)["keep"] == "me"


def test_G2_khong_de_lai_file_tam(tmp_path: Path) -> None:
    target = tmp_path / "manifest.jsonl"
    write_jsonl_atomic([{"table_ref": "DOC|1"}], target)

    def bad_rows():
        yield {"table_ref": "DOC|2"}
        raise RuntimeError("loi")

    with pytest.raises(RuntimeError):
        write_jsonl_atomic(bad_rows(), target)

    leftovers = [p.name for p in tmp_path.iterdir() if p.name.startswith(".")]
    assert not leftovers, f"con file tam: {leftovers}"


def test_G3_atomic_tra_ve_so_dong(tmp_path: Path) -> None:
    n = write_jsonl_atomic(({"table_ref": f"D|{i}"} for i in range(33)),
                           tmp_path / "m.jsonl")
    assert n == 33


# ── manifest key ──────────────────────────────────────────


def test_table_ref_la_khoa_duy_nhat_khong_phai_filename(tmp_path: Path) -> None:
    """Hai doc_id khac nhau co the sinh cung filename sau khi sanitize —
    khoa phai la table_ref, khong duoc la filename."""
    assert CsvWriter.make_filename("A/B", 1) == CsvWriter.make_filename("A_B", 1)
    w = _writer(tmp_path)
    w.write(_table("A/B", 1), _df(), ticker="AAA", year=2020)
    w.write(_table("A_B", 1), _df(), ticker="AAA", year=2020)
    target = tmp_path / "m.jsonl"
    w.write_manifest(target, mode="full")
    refs = [r["table_ref"] for r in read_jsonl(target)]
    assert refs == ["A/B|1", "A_B|1"], "hai bang khac nhau phai giu hai entry"
