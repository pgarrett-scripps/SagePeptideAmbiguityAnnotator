"""Tests for the ``--folder`` and ``--data`` input modes, and for argument validation.

``tests/test_cli.py`` covers the ``--results``/``--fragments`` pair that predates these
modes; it is unchanged and must keep passing, since the entry point now resolves to
``cli.main``.
"""

import logging

import pandas as pd
import pytest

from sage_peptide_ambiguity_annotator.cli import (
    Config,
    determine_output_path,
    get_folders_to_process,
    main,
    parse_arguments,
)

EXPECTED_PSM_1 = "(?PE)PTI(?DE)"


def _run_cli(monkeypatch, argv):
    monkeypatch.setattr("sys.argv", ["sage-annotate", *argv])
    return main()


def _read(path):
    return pd.read_csv(path, sep="\t")


# --- --folder ---------------------------------------------------------------------


def test_folder_mode_writes_the_requested_output(monkeypatch, tmp_path, sage_folder):
    output = tmp_path / "annotated.sage.tsv"

    exit_code = _run_cli(monkeypatch, ["--folder", str(sage_folder), "--output", str(output)])

    assert exit_code == 0
    out = _read(output)
    assert len(out) == 5
    assert out["ambiguity_sequence"].notna().all()
    assert out.loc[out["psm_id"] == 1, "ambiguity_sequence"].item() == EXPECTED_PSM_1


def test_folder_mode_short_flags(monkeypatch, tmp_path, sage_folder):
    output = tmp_path / "annotated.sage.tsv"

    exit_code = _run_cli(monkeypatch, ["-f", str(sage_folder), "-o", str(output)])

    assert exit_code == 0
    assert output.exists()


def test_folder_mode_matches_the_explicit_file_pair(
    monkeypatch, tmp_path, sage_folder, results_path, fragments_path
):
    """--folder and --results/--fragments produce the same annotation."""
    from_folder = tmp_path / "from_folder.sage.tsv"
    from_pair = tmp_path / "from_pair.sage.tsv"

    assert _run_cli(monkeypatch, ["-f", str(sage_folder), "-o", str(from_folder)]) == 0
    assert (
        _run_cli(
            monkeypatch,
            ["-r", results_path, "-frag", fragments_path, "-o", str(from_pair)],
        )
        == 0
    )

    pd.testing.assert_frame_equal(_read(from_folder), _read(from_pair))


def test_folder_mode_reports_a_missing_folder(monkeypatch, tmp_path):
    exit_code = _run_cli(
        monkeypatch,
        ["--folder", str(tmp_path / "nope"), "--output", str(tmp_path / "out.sage.tsv")],
    )

    assert exit_code == 1


def test_folder_mode_reports_an_incomplete_folder(monkeypatch, tmp_path, sage_folder):
    (sage_folder / "matched_fragments.sage.tsv").unlink()

    exit_code = _run_cli(
        monkeypatch,
        ["--folder", str(sage_folder), "--output", str(tmp_path / "out.sage.tsv")],
    )

    assert exit_code == 1


# --- --data -----------------------------------------------------------------------


def test_data_mode_writes_one_output_per_subfolder(monkeypatch, sage_data_dir):
    exit_code = _run_cli(
        monkeypatch, ["--data", str(sage_data_dir), "--output", "annotated.sage.tsv"]
    )

    assert exit_code == 0
    for name in ("run_a", "run_b"):
        output = sage_data_dir / name / "annotated.sage.tsv"
        assert output.exists(), f"no output written in {name}"
        out = _read(output)
        assert out.loc[out["psm_id"] == 1, "ambiguity_sequence"].item() == EXPECTED_PSM_1


def test_data_mode_uses_only_the_output_file_name(monkeypatch, tmp_path, sage_data_dir):
    """A full --output path is reduced to its file name inside each sub-folder."""
    exit_code = _run_cli(
        monkeypatch,
        ["-d", str(sage_data_dir), "-o", str(tmp_path / "elsewhere" / "annotated.sage.tsv")],
    )

    assert exit_code == 0
    assert not (tmp_path / "elsewhere").exists()
    assert (sage_data_dir / "run_a" / "annotated.sage.tsv").exists()


def test_data_mode_skips_a_subfolder_without_sage_output(
    monkeypatch, sage_data_dir, caplog
):
    (sage_data_dir / "not_a_search").mkdir()

    with caplog.at_level(logging.WARNING):
        exit_code = _run_cli(
            monkeypatch, ["--data", str(sage_data_dir), "--output", "annotated.sage.tsv"]
        )

    assert exit_code == 0
    assert not (sage_data_dir / "not_a_search" / "annotated.sage.tsv").exists()
    assert (sage_data_dir / "run_a" / "annotated.sage.tsv").exists()
    assert any("Skipping folder" in record.message for record in caplog.records)


def test_data_mode_ignores_loose_files(monkeypatch, sage_data_dir):
    (sage_data_dir / "notes.txt").write_text("not a folder")

    exit_code = _run_cli(
        monkeypatch, ["--data", str(sage_data_dir), "--output", "annotated.sage.tsv"]
    )

    assert exit_code == 0


def test_data_mode_with_no_subfolders(monkeypatch, tmp_path):
    empty = tmp_path / "data"
    empty.mkdir()

    exit_code = _run_cli(monkeypatch, ["--data", str(empty), "--output", "annotated.sage.tsv"])

    assert exit_code == 1


def test_data_mode_with_no_valid_subfolders(monkeypatch, tmp_path):
    data = tmp_path / "data"
    (data / "empty_one").mkdir(parents=True)
    (data / "empty_two").mkdir()

    exit_code = _run_cli(monkeypatch, ["--data", str(data), "--output", "annotated.sage.tsv"])

    assert exit_code == 1


def test_data_mode_reports_a_missing_directory(monkeypatch, tmp_path):
    exit_code = _run_cli(
        monkeypatch, ["--data", str(tmp_path / "nope"), "--output", "annotated.sage.tsv"]
    )

    assert exit_code == 1


# --- argument validation ----------------------------------------------------------


def test_an_input_mode_is_required():
    with pytest.raises(SystemExit) as excinfo:
        parse_arguments(["--output", "out.sage.tsv"])

    assert excinfo.value.code == 2


@pytest.mark.parametrize(
    "argv",
    [
        ["--folder", "f", "--results", "r"],
        ["--folder", "f", "--data", "d"],
        ["--results", "r", "--data", "d"],
    ],
)
def test_input_modes_are_mutually_exclusive(argv):
    with pytest.raises(SystemExit) as excinfo:
        parse_arguments([*argv, "--output", "out.sage.tsv"])

    assert excinfo.value.code == 2


def test_results_requires_fragments():
    with pytest.raises(SystemExit) as excinfo:
        parse_arguments(["--results", "r", "--output", "out.sage.tsv"])

    assert excinfo.value.code == 2


def test_fragments_requires_results():
    with pytest.raises(SystemExit) as excinfo:
        parse_arguments(["--fragments", "f", "--output", "out.sage.tsv"])

    assert excinfo.value.code == 2


def test_output_is_still_required():
    with pytest.raises(SystemExit) as excinfo:
        parse_arguments(["--folder", "f"])

    assert excinfo.value.code == 2


def test_short_and_long_flags_parse_to_the_same_namespace():
    long_form = parse_arguments(
        [
            "--results",
            "r",
            "--fragments",
            "f",
            "--output",
            "o",
            "--mass_error_type",
            "Da",
            "--mass_error_value",
            "0.02",
            "--mass_shift",
            "--verbose",
        ]
    )
    short_form = parse_arguments(
        ["-r", "r", "-frag", "f", "-o", "o", "-type", "Da", "-value", "0.02", "-shift", "-V"]
    )

    assert vars(long_form) == vars(short_form)
    assert short_form.mass_error_type == "Da"
    assert short_form.mass_error_value == pytest.approx(0.02)
    assert short_form.mass_shift is True
    assert short_form.verbose is True


def test_defaults_match_the_documented_values():
    args = parse_arguments(["--folder", "f", "--output", "o"])

    assert args.mass_error_type == "ppm"
    assert args.mass_error_value == pytest.approx(50.0)
    assert args.mass_shift is False
    assert args.verbose is False
    assert args.fragments is None


# --- helpers ----------------------------------------------------------------------


def test_determine_output_path_single_folder(sage_folder):
    args = parse_arguments(["--folder", str(sage_folder), "--output", "/tmp/out.sage.tsv"])

    assert determine_output_path(args, str(sage_folder)) == "/tmp/out.sage.tsv"


def test_determine_output_path_data_mode(sage_data_dir):
    args = parse_arguments(["--data", str(sage_data_dir), "--output", "/tmp/out.sage.tsv"])
    folder = str(sage_data_dir / "run_a")

    assert determine_output_path(args, folder) == f"{folder}/out.sage.tsv"


def test_get_folders_to_process_is_ordered(sage_data_dir):
    args = parse_arguments(["--data", str(sage_data_dir), "--output", "annotated.sage.tsv"])

    folders = get_folders_to_process(args)

    assert [f[0].rsplit("/", 1)[-1] for f in folders] == ["run_a", "run_b"]


def test_config_defaults():
    config = Config()

    assert config.mass_error_type == "ppm"
    assert config.mass_error_value == pytest.approx(50.0)
    assert config.use_mass_shift is False
    assert config.verbose is False
    assert config.output is None
