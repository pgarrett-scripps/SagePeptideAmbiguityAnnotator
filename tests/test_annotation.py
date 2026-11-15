"""Tests for the annotation pipeline against the committed example input.

The example in ``tests/data`` is five PSMs over two peptides and covers every branch the
annotator has: a peptide ambiguous at both termini, one whose mass shift localizes to a
single interval, one whose shift cannot be localized and becomes a labile modification, a
peptide carrying a modification on the input sequence, and a PSM with no matched fragments
at all.
"""

import pandas as pd
import pytest

from sage_peptide_ambiguity_annotator.main import (
    count_fragments_per_amino_acid,
    map_fragment_type_to_aa_position,
    process_psm_data,
    read_input_files,
    save_output,
)


def _annotate(results_path, fragments_path, *, use_mass_shift):
    results_df, fragments_df = read_input_files(results_path, fragments_path)
    return process_psm_data(
        results_df,
        fragments_df,
        mass_error_type="ppm",
        mass_error_value=50.0,
        use_mass_shift=use_mass_shift,
    )


def test_read_input_files_derives_stripped_peptide(results_path, fragments_path):
    """The example results file has no ``stripped_peptide`` column; it is derived."""
    results_df, fragments_df = read_input_files(results_path, fragments_path)

    assert "stripped_peptide" in results_df.columns
    assert list(results_df["stripped_peptide"]) == [
        "PEPTIDE",
        "PEPTIDE",
        "PEPTIDE",
        "CPEPTIDE",
        "PEPTIDE",
    ]
    assert len(results_df) == 5
    assert len(fragments_df) == 20


def test_read_input_files_rejects_missing_results_column(tmp_path, fragments_path):
    bad = tmp_path / "results.sage.tsv"
    bad.write_text("psm_id\tpeptide\texpmass\n1\tPEPTIDE\t799.35972\n")

    with pytest.raises(ValueError, match="Missing required columns in results file"):
        read_input_files(str(bad), fragments_path)


def test_read_input_files_rejects_missing_fragments_column(results_path, tmp_path):
    bad = tmp_path / "matched_fragments.sage.tsv"
    bad.write_text("psm_id\tfragment_type\n1\tb\n")

    with pytest.raises(ValueError, match="Missing required columns in fragments file"):
        read_input_files(results_path, str(bad))


def test_read_input_files_rejects_unsupported_extension(tmp_path, fragments_path):
    bad = tmp_path / "results.sage.csv"
    bad.write_text("psm_id,peptide\n1,PEPTIDE\n")

    with pytest.raises(ValueError, match="Unsupported file format for results"):
        read_input_files(str(bad), fragments_path)


@pytest.mark.parametrize(
    ("fragment_type", "ordinal", "expected"),
    [
        ("b", 1, 0),
        ("b", 3, 2),
        ("a", 2, 1),
        ("c", 7, 6),
        ("y", 1, 6),
        ("y", 3, 4),
        ("z", 7, 0),
        ("?", 2, -1),
    ],
)
def test_map_fragment_type_to_aa_position(fragment_type, ordinal, expected):
    """Forward ions index from the N terminus, reverse ions from the C terminus."""
    assert map_fragment_type_to_aa_position("PEPTIDE", fragment_type, ordinal) == expected


def test_count_fragments_per_amino_acid(results_path, fragments_path):
    results_df, fragments_df = read_input_files(results_path, fragments_path)
    counts = count_fragments_per_amino_acid(results_df, fragments_df)

    _, forward, reverse = counts[1]
    assert forward == [0, 1, 1, 0, 0, 0, 0]
    assert reverse == [0, 0, 0, 0, 1, 1, 0]

    # PSM 5 has no matched fragments, so every position reads zero coverage.
    _, forward, reverse = counts[5]
    assert forward == [0] * 7
    assert reverse == [0] * 7


def test_annotation_without_mass_shift(results_path, fragments_path):
    out = _annotate(results_path, fragments_path, use_mass_shift=False)
    annotated = dict(zip(out["psm_id"], out["ambiguity_sequence"], strict=True))

    assert annotated == {
        1: "(?PE)PTI(?DE)",
        2: "(?PE)PT(?ID)E",
        3: "(?PE)PTI(?DE)",
        4: "(?C[+57.0215]P)E(?PT)I(?DE)",
        5: "(?PEPTIDE)",
    }
    # With --mass_shift off, no row carries a shift.
    assert out["mass_shift"].isna().all()


def test_annotation_with_mass_shift(results_path, fragments_path):
    out = _annotate(results_path, fragments_path, use_mass_shift=True)
    annotated = dict(zip(out["psm_id"], out["ambiguity_sequence"], strict=True))

    assert annotated == {
        # Unshifted PSMs are unchanged: the 0 ppm difference is inside the 50 ppm threshold.
        1: "(?PE)PTI(?DE)",
        # A 100 Da shift that localizes to one ambiguous interval.
        2: "(?PE)PT(?ID)[+100.0]E",
        # The same shift when forward and reverse coverage overlap: a labile modification.
        3: "{+100.0}(?PE)PTI(?DE)",
        4: "(?C[+57.0215]P)E(?PT)I(?DE)",
        5: "(?PEPTIDE)",
    }
    shifts = dict(zip(out["psm_id"], out["mass_shift"], strict=True))
    assert shifts[2] == pytest.approx(100.0, abs=1e-6)
    assert shifts[3] == pytest.approx(100.0, abs=1e-6)
    for psm_id in (1, 4, 5):
        assert shifts[psm_id] is None or pd.isna(shifts[psm_id])


def test_annotation_is_never_silently_empty(results_path, fragments_path):
    """Guards the failure mode that the peptacular 5 port fixed.

    Every annotation error is caught per PSM and recorded as ``None``, so an incompatible
    peptacular made the tool exit 0 with an all-null ``ambiguity_sequence`` column. An
    all-null column is therefore always a bug, never a valid result for this input.
    """
    for use_mass_shift in (False, True):
        out = _annotate(results_path, fragments_path, use_mass_shift=use_mass_shift)
        assert out["ambiguity_sequence"].notna().all()


def test_output_columns_are_added_to_the_input(results_path, fragments_path):
    results_df, _ = read_input_files(results_path, fragments_path)
    out = _annotate(results_path, fragments_path, use_mass_shift=True)

    assert set(results_df.columns) <= set(out.columns)
    assert out.columns[-2:].tolist() == ["ambiguity_sequence", "mass_shift"]
    assert len(out) == len(results_df)


@pytest.mark.parametrize("suffix", [".tsv", ".parquet"])
def test_save_output_round_trip(results_path, fragments_path, tmp_path, suffix):
    out = _annotate(results_path, fragments_path, use_mass_shift=True)
    path = tmp_path / f"annotated_results.sage{suffix}"

    save_output(out, str(path))

    assert path.exists()
    if suffix == ".tsv":
        reloaded = pd.read_csv(path, sep="\t")
    else:
        reloaded = pd.read_parquet(path)
    assert list(reloaded["ambiguity_sequence"]) == list(out["ambiguity_sequence"])


def test_save_output_rejects_unsupported_extension(results_path, fragments_path, tmp_path):
    out = _annotate(results_path, fragments_path, use_mass_shift=False)

    with pytest.raises(ValueError, match="Unsupported output file format"):
        save_output(out, str(tmp_path / "annotated_results.csv"))
