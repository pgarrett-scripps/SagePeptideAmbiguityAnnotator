"""Pins the annotation output across the peptacular 2.5.0 -> 5.x port.

Release 1.0.1 pinned ``peptacular==2.5.0``. Both goldens in ``tests/data`` were produced by
running this package's own pipeline over ``tests/data/results.sage.tsv`` and
``tests/data/matched_fragments.sage.tsv``, once against each peptacular version:

* ``expected_peptacular_2_5_0.tsv`` — the pipeline as released in 1.0.1, on peptacular 2.5.0.
* ``expected_peptacular_5.tsv`` — the ported pipeline, on peptacular 5.x.

The port changes the output in exactly one way: peptacular 5 writes an explicit ``+`` on
positive mass values, both for the mass shift it places and for modifications already
present on the input peptide. Ambiguity intervals, the localization decision, the labile
fallback and the ``mass_shift`` column are all unchanged. These tests assert both halves of
that statement, so any future change to either is a test failure rather than a silent
change to a published output column.
"""

import re

import pandas as pd
import pytest

from sage_peptide_ambiguity_annotator.main import process_psm_data, read_input_files

#: peptacular 2.5.0 wrote "[57.0215]"; peptacular 5 writes "[+57.0215]". Negative values
#: already carried their sign in both versions, so only a leading digit needs the "+".
_UNSIGNED_POSITIVE_MASS = re.compile(r"([\[{])(\d)")


def _add_explicit_plus(sequence: str) -> str:
    return _UNSIGNED_POSITIVE_MASS.sub(r"\1+\2", sequence)


def _run(results_path, fragments_path) -> pd.DataFrame:
    results_df, fragments_df = read_input_files(results_path, fragments_path)
    rows = []
    for use_mass_shift in (False, True):
        out = process_psm_data(
            results_df,
            fragments_df,
            mass_error_type="ppm",
            mass_error_value=50.0,
            use_mass_shift=use_mass_shift,
        )
        for _, row in out.iterrows():
            rows.append(
                {
                    "use_mass_shift": use_mass_shift,
                    "psm_id": row["psm_id"],
                    "peptide": row["peptide"],
                    "stripped_peptide": row["stripped_peptide"],
                    "ambiguity_sequence": row["ambiguity_sequence"],
                    "mass_shift": row["mass_shift"],
                }
            )
    frame = pd.DataFrame(rows)
    frame["mass_shift"] = frame["mass_shift"].astype(float)
    return frame


def test_pipeline_reproduces_the_peptacular_5_golden(
    results_path, fragments_path, golden_5
):
    """The ported pipeline output is byte-for-byte the recorded peptacular 5 output."""
    actual = _run(results_path, fragments_path)

    pd.testing.assert_frame_equal(
        actual.reset_index(drop=True),
        golden_5.reset_index(drop=True),
        check_dtype=False,
    )


def test_port_changes_only_the_explicit_plus_sign(golden_2_5_0, golden_5):
    """Signing every positive mass in the 2.5.0 output reproduces the 5.x output."""
    assert len(golden_2_5_0) == len(golden_5)

    ported = golden_2_5_0["ambiguity_sequence"].map(_add_explicit_plus)
    pd.testing.assert_series_equal(
        ported.reset_index(drop=True),
        golden_5["ambiguity_sequence"].reset_index(drop=True),
        check_names=False,
    )


def test_port_leaves_the_mass_shift_column_unchanged(golden_2_5_0, golden_5):
    pd.testing.assert_series_equal(
        golden_2_5_0["mass_shift"].reset_index(drop=True),
        golden_5["mass_shift"].reset_index(drop=True),
        check_names=False,
    )


def test_port_leaves_ambiguity_intervals_unchanged(golden_2_5_0, golden_5):
    """Stripping every mass value leaves identical interval structure on both versions."""

    def intervals_only(sequence: str) -> str:
        return re.sub(r"[\[{][^\]}]*[\]}]", "", sequence)

    before = golden_2_5_0["ambiguity_sequence"].map(intervals_only)
    after = golden_5["ambiguity_sequence"].map(intervals_only)
    pd.testing.assert_series_equal(
        before.reset_index(drop=True),
        after.reset_index(drop=True),
        check_names=False,
    )


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("(?PE)PTI(?DE)", "(?PE)PTI(?DE)"),
        ("(?PE)PT(?ID)[100.0]E", "(?PE)PT(?ID)[+100.0]E"),
        ("{100.0}(?PE)PTI(?DE)", "{+100.0}(?PE)PTI(?DE)"),
        ("(?C[57.0215]P)E(?PT)I(?DE)", "(?C[+57.0215]P)E(?PT)I(?DE)"),
        ("PEPTIDE-[-0.984]", "PEPTIDE-[-0.984]"),
    ],
)
def test_add_explicit_plus_helper(before, after):
    """The helper the parity test relies on, pinned on its own."""
    assert _add_explicit_plus(before) == after
