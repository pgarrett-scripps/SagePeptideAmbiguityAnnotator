"""The ``main`` module path is kept as a shim; this pins what it must keep exporting.

Release 1.0.1 documented ``from sage_peptide_ambiguity_annotator.main import ...`` as the
Python API, and ``streamlit_app.py`` and the ``sage-annotate`` entry point used it. The
pipeline now lives in ``utils`` and the CLI in ``cli``, so ``main`` re-exports both.
"""

import sage_peptide_ambiguity_annotator.cli as cli
import sage_peptide_ambiguity_annotator.main as main_module
import sage_peptide_ambiguity_annotator.utils as utils

PIPELINE_NAMES = [
    "count_fragments_per_amino_acid",
    "create_output_dataframe",
    "find_sage_files",
    "map_fragment_type_to_aa_position",
    "process_psm_data",
    "read_input_files",
    "save_output",
]

CLI_NAMES = ["main", "parse_arguments", "setup_logging"]


def test_main_reexports_every_name_1_0_1_exported():
    for name in [*PIPELINE_NAMES, *CLI_NAMES]:
        assert hasattr(main_module, name), f"main.{name} is gone"


def test_main_reexports_are_the_same_objects():
    for name in PIPELINE_NAMES:
        assert getattr(main_module, name) is getattr(utils, name)
    for name in CLI_NAMES:
        assert getattr(main_module, name) is getattr(cli, name)


def test_entry_point_target_is_callable():
    """``sage-annotate`` resolves to ``cli:main``, and ``main:main`` is the same function."""
    assert callable(cli.main)
    assert main_module.main is cli.main
