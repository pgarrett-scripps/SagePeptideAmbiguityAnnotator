"""Tests for ``find_sage_files``, the folder discovery behind ``--folder`` and ``--data``."""

import pytest

from sage_peptide_ambiguity_annotator.utils import find_sage_files


def test_finds_both_files(sage_folder):
    results_path, fragments_path = find_sage_files(str(sage_folder))

    assert results_path == str(sage_folder / "results.sage.tsv")
    assert fragments_path == str(sage_folder / "matched_fragments.sage.tsv")


def test_finds_parquet_as_well_as_tsv(tmp_path):
    folder = tmp_path / "search"
    folder.mkdir()
    (folder / "results.sage.parquet").write_bytes(b"")
    (folder / "matched_fragments.sage.parquet").write_bytes(b"")

    results_path, fragments_path = find_sage_files(str(folder))

    assert results_path.endswith("results.sage.parquet")
    assert fragments_path.endswith("matched_fragments.sage.parquet")


def test_missing_folder(tmp_path):
    with pytest.raises(FileNotFoundError, match="Folder not found"):
        find_sage_files(str(tmp_path / "nope"))


def test_path_is_a_file(sage_folder):
    with pytest.raises(ValueError, match="Path is not a directory"):
        find_sage_files(str(sage_folder / "results.sage.tsv"))


def test_missing_results_file(sage_folder):
    (sage_folder / "results.sage.tsv").unlink()

    with pytest.raises(FileNotFoundError, match=r"No results\.sage\.\* file found"):
        find_sage_files(str(sage_folder))


def test_missing_fragments_file(sage_folder):
    (sage_folder / "matched_fragments.sage.tsv").unlink()

    with pytest.raises(
        FileNotFoundError, match=r"No matched_fragments\.sage\.\* file found"
    ):
        find_sage_files(str(sage_folder))


def test_ambiguous_results_file(sage_folder):
    """A folder holding both a .tsv and a .parquet is refused rather than guessed at."""
    (sage_folder / "results.sage.parquet").write_bytes(b"")

    with pytest.raises(ValueError, match=r"Multiple results\.sage\.\* files found"):
        find_sage_files(str(sage_folder))


def test_ambiguous_fragments_file(sage_folder):
    (sage_folder / "matched_fragments.sage.parquet").write_bytes(b"")

    with pytest.raises(
        ValueError, match=r"Multiple matched_fragments\.sage\.\* files found"
    ):
        find_sage_files(str(sage_folder))


def test_unrelated_files_are_ignored(sage_folder):
    (sage_folder / "results.sage.csv").write_text("")
    (sage_folder / "peptide.tsv").write_text("")
    (sage_folder / "annotated_results.sage.tsv").write_text("")

    results_path, _ = find_sage_files(str(sage_folder))

    assert results_path.endswith("results.sage.tsv")
