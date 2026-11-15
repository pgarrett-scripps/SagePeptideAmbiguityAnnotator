"""End-to-end tests for the ``sage-annotate`` command line entry point."""

import logging

import pandas as pd
import pytest

from sage_peptide_ambiguity_annotator.main import main


def _run_cli(monkeypatch, argv):
    monkeypatch.setattr("sys.argv", ["sage-annotate", *argv])
    return main()


def test_cli_writes_annotated_output(monkeypatch, tmp_path, results_path, fragments_path):
    output = tmp_path / "annotated_results.sage.tsv"

    exit_code = _run_cli(
        monkeypatch,
        ["--results", results_path, "--fragments", fragments_path, "--output", str(output)],
    )

    assert exit_code == 0
    out = pd.read_csv(output, sep="\t")
    assert out["ambiguity_sequence"].notna().all()
    assert out.loc[out["psm_id"] == 1, "ambiguity_sequence"].item() == "(?PE)PTI(?DE)"


def test_cli_mass_shift_flag(monkeypatch, tmp_path, results_path, fragments_path):
    output = tmp_path / "annotated_results.sage.tsv"

    exit_code = _run_cli(
        monkeypatch,
        [
            "--results",
            results_path,
            "--fragments",
            fragments_path,
            "--output",
            str(output),
            "--mass_error_type",
            "ppm",
            "--mass_error_value",
            "50.0",
            "--mass_shift",
        ],
    )

    assert exit_code == 0
    out = pd.read_csv(output, sep="\t")
    assert out.loc[out["psm_id"] == 2, "ambiguity_sequence"].item() == "(?PE)PT(?ID)[+100.0]E"
    assert out.loc[out["psm_id"] == 3, "ambiguity_sequence"].item() == "{+100.0}(?PE)PTI(?DE)"


def test_cli_creates_missing_output_directory(
    monkeypatch, tmp_path, results_path, fragments_path
):
    output = tmp_path / "nested" / "dir" / "annotated_results.sage.tsv"

    exit_code = _run_cli(
        monkeypatch,
        ["--results", results_path, "--fragments", fragments_path, "--output", str(output)],
    )

    assert exit_code == 0
    assert output.exists()


def test_cli_reports_a_missing_input_file(monkeypatch, tmp_path, fragments_path):
    exit_code = _run_cli(
        monkeypatch,
        [
            "--results",
            str(tmp_path / "does_not_exist.sage.tsv"),
            "--fragments",
            fragments_path,
            "--output",
            str(tmp_path / "out.sage.tsv"),
        ],
    )

    assert exit_code == 1


def test_cli_requires_output(monkeypatch, results_path, fragments_path):
    with pytest.raises(SystemExit) as excinfo:
        _run_cli(monkeypatch, ["--results", results_path, "--fragments", fragments_path])

    assert excinfo.value.code == 2


def test_cli_fails_when_no_psm_is_annotated(
    monkeypatch, tmp_path, results_path, fragments_path, caplog
):
    def fail(*args, **kwargs):
        raise ValueError("no fragment ion matches")

    monkeypatch.setattr(
        "sage_peptide_ambiguity_annotator.utils.pt.annotate_ambiguity", fail
    )
    caplog.set_level(logging.INFO)

    output = tmp_path / "annotated_results.sage.tsv"

    exit_code = _run_cli(
        monkeypatch,
        ["--results", results_path, "--fragments", fragments_path, "--output", str(output)],
    )

    assert exit_code == 1
    out = pd.read_csv(output, sep="\t")
    assert out["ambiguity_sequence"].isna().all()
    assert "Annotated 0 of" in caplog.text
