"""Shared fixtures for the SagePeptideAmbiguityAnnotator test suite."""

import shutil
from pathlib import Path

import pandas as pd
import pytest

DATA_DIR = Path(__file__).parent / "data"

RESULTS_TSV = DATA_DIR / "results.sage.tsv"
FRAGMENTS_TSV = DATA_DIR / "matched_fragments.sage.tsv"

#: Annotation output recorded from peptacular 2.5.0, the version pinned by release 1.0.1.
GOLDEN_2_5_0 = DATA_DIR / "expected_peptacular_2_5_0.tsv"

#: Annotation output recorded from peptacular 5.x, the version this package now requires.
GOLDEN_5 = DATA_DIR / "expected_peptacular_5.tsv"


def _read_golden(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    # mass_shift is NaN for every row where no shift was localized; keep it as a float
    # column so NaN comparisons stay comparable between the two goldens.
    df["mass_shift"] = df["mass_shift"].astype(float)
    return df


@pytest.fixture
def results_path() -> str:
    return str(RESULTS_TSV)


@pytest.fixture
def fragments_path() -> str:
    return str(FRAGMENTS_TSV)


@pytest.fixture
def golden_2_5_0() -> pd.DataFrame:
    return _read_golden(GOLDEN_2_5_0)


@pytest.fixture
def golden_5() -> pd.DataFrame:
    return _read_golden(GOLDEN_5)


def _make_sage_folder(folder: Path) -> Path:
    """Copy the committed example into ``folder`` under the names Sage writes."""
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy(RESULTS_TSV, folder / "results.sage.tsv")
    shutil.copy(FRAGMENTS_TSV, folder / "matched_fragments.sage.tsv")
    return folder


@pytest.fixture
def sage_folder(tmp_path) -> Path:
    """A single Sage output folder, as ``--folder`` expects."""
    return _make_sage_folder(tmp_path / "search")


@pytest.fixture
def sage_data_dir(tmp_path) -> Path:
    """A directory of two Sage output folders, as ``--data`` expects."""
    data = tmp_path / "data"
    _make_sage_folder(data / "run_a")
    _make_sage_folder(data / "run_b")
    return data
