"""
Backwards-compatible module path.

The pipeline moved to :mod:`sage_peptide_ambiguity_annotator.utils` and the command line
interface to :mod:`sage_peptide_ambiguity_annotator.cli`. Everything that
``sage_peptide_ambiguity_annotator.main`` exported before the split is re-exported here,
so existing imports and the ``sage_peptide_ambiguity_annotator.main:main`` entry point
keep working. New code should import from ``utils`` or ``cli`` directly.
"""

from .cli import main, parse_arguments, setup_logging
from .utils import (
    count_fragments_per_amino_acid,
    create_output_dataframe,
    find_sage_files,
    map_fragment_type_to_aa_position,
    process_psm_data,
    read_input_files,
    save_output,
)

__all__ = [
    "count_fragments_per_amino_acid",
    "create_output_dataframe",
    "find_sage_files",
    "main",
    "map_fragment_type_to_aa_position",
    "parse_arguments",
    "process_psm_data",
    "read_input_files",
    "save_output",
    "setup_logging",
]
