"""
SagePeptideAmbiguityAnnotator main module.

This module processes peptide spectrum matches (PSMs) from Sage search engine output
and annotates peptides with ambiguity information based on fragment ion coverage.
"""

import os
import sys
import logging
import pandas as pd
import argparse
import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import peptacular as pt


def setup_logging(verbose: bool = False) -> None:
    """
    Set up logging configuration.

    Args:
        verbose: Whether to show verbose (DEBUG) logs
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=log_level, format=log_format)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process Sage PSM and fragment data to annotate peptide ambiguity",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to the results.sage.parquet or results.sage.tsv file",
    )
    parser.add_argument(
        "--fragments",
        type=str,
        required=True,
        help="Path to the matched_fragments.sage.parquet or matched_fragments.tsv file",
    )

    # Mass error options
    parser.add_argument(
        "--mass_error_type",
        type=str,
        default="ppm",
        choices=["ppm", "Da"],
        help="Type of mass error (ppm or Da)",
    )
    parser.add_argument(
        "--mass_error_value", type=float, default=50.0, help="Value of mass error"
    )
    parser.add_argument(
        "--mass_shift", action="store_true", help="Turn on mass shift annotation"
    )

    # Output options
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the output file (.parquet or .tsv)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    return parser.parse_args()


def read_input_files(
    results_path: str, fragments_path: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read the results and fragments parquet or tsv files.

    Args:
        results_path: Path to the results.sage.parquet or results.sage.tsv file
        fragments_path: Path to the matched_fragments.sage.parquet or matched_fragments.sage.tsv file

    Returns:
        Tuple of (results_df, fragments_df)

    Raises:
        FileNotFoundError: If input files don't exist
        ValueError: If file format is unsupported or required columns are missing
    """
    logger = logging.getLogger(__name__)

    # Check that files exist
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found: {results_path}")

    if not os.path.exists(fragments_path):
        raise FileNotFoundError(f"Fragments file not found: {fragments_path}")

    # Read results file
    logger.info(f"Reading results file from {results_path}")
    if results_path.endswith(".parquet"):
        results_df = pd.read_parquet(results_path)
    elif results_path.endswith(".tsv"):
        results_df = pd.read_csv(results_path, sep="\t")
    else:
        raise ValueError(
            "Unsupported file format for results. Only .parquet and .tsv are supported."
        )

    # Check required columns in results
    required_results_cols = [
        "psm_id",
        "peptide",
        "stripped_peptide",
        "expmass",
        "calcmass",
    ]
    missing_cols = [
        col for col in required_results_cols if col not in results_df.columns
    ]
    if missing_cols:
        raise ValueError(
            f"Missing required columns in results file: {', '.join(missing_cols)}"
        )

    # Read fragments file
    logger.info(f"Reading fragments file from {fragments_path}")
    if fragments_path.endswith(".parquet"):
        fragments_df = pd.read_parquet(fragments_path)
    elif fragments_path.endswith(".tsv"):
        fragments_df = pd.read_csv(fragments_path, sep="\t")
    else:
        raise ValueError(
            "Unsupported file format for fragments. Only .parquet and .tsv are supported."
        )

    # Check required columns in fragments
    required_fragments_cols = ["psm_id", "fragment_type", "fragment_ordinals"]
    missing_cols = [
        col for col in required_fragments_cols if col not in fragments_df.columns
    ]
    if missing_cols:
        raise ValueError(
            f"Missing required columns in fragments file: {', '.join(missing_cols)}"
        )

    logger.info(f"Read {len(results_df)} PSMs and {len(fragments_df)} fragment ions")
    return results_df, fragments_df


def map_fragment_type_to_aa_position(
    peptide: str, fragment_type: str, fragment_ordinal: int
) -> int:
    """
    Map fragment ions to amino acid positions.

    Args:
        peptide: The peptide sequence
        fragment_type: The type of fragment (a, b, c, x, y, z, etc.)
        fragment_ordinal: The ordinal position of the fragment

    Returns:
        The position in the peptide (0-indexed) that the fragment corresponds to
    """
    # Strip any modifications from the peptide
    stripped_peptide = "".join(c for c in peptide if c.isalpha())

    # Forward ion types (N-terminal fragments)
    if fragment_type.startswith(("a", "b", "c")):
        # a, b, c ions represent N-terminal fragments
        # The ordinal - 1 is the position (0-indexed)
        return fragment_ordinal - 1

    # Reverse ion types (C-terminal fragments)
    elif fragment_type.startswith(("x", "y", "z")):
        # x, y, z ions represent C-terminal fragments
        # Count from the end of the peptide
        return len(stripped_peptide) - fragment_ordinal

    # Handle other fragment types if needed
    return -1  # Return -1 for unsupported fragment types


def count_fragments_per_amino_acid(
    results_df: pd.DataFrame, fragments_df: pd.DataFrame
) -> Dict[int, Tuple[List[int], List[int], List[int]]]:
    """
    Count the number of fragment ions matching to each amino acid in the peptide,
    separating forward ions (a, b, c) and reverse ions (x, y, z).

    Args:
        results_df: DataFrame containing PSM data
        fragments_df: DataFrame containing fragment ion data

    Returns:
        Dictionary mapping psm_id to a tuple of (all_fragment_counts, forward_fragment_counts, reverse_fragment_counts)
    """
    # Initialize dictionary to store results
    fragment_counts_by_psm = {}

    # Group fragments by psm_id
    grouped_fragments = fragments_df.groupby("psm_id")

    # Process each PSM
    for _, psm_row in results_df.iterrows():
        psm_id = psm_row["psm_id"]
        peptide = psm_row["stripped_peptide"]

        # Skip if peptide is missing or not a string
        if not isinstance(peptide, str) or not peptide:
            continue

        # Initialize arrays with zeros for each amino acid in the peptide
        all_fragment_counts = [0] * len(peptide)
        forward_fragment_counts = [0] * len(peptide)  # a, b, c ions
        reverse_fragment_counts = [0] * len(peptide)  # x, y, z ions

        # Get fragments for this PSM
        if psm_id in grouped_fragments.groups:
            psm_fragments = grouped_fragments.get_group(psm_id)

            # Process each fragment
            for _, fragment in psm_fragments.iterrows():
                fragment_type = fragment["fragment_type"]
                fragment_ordinal = fragment["fragment_ordinals"]

                # Map fragment to amino acid position
                aa_position = map_fragment_type_to_aa_position(
                    peptide, fragment_type, fragment_ordinal
                )

                # Increment count if position is valid
                if 0 <= aa_position < len(peptide):
                    all_fragment_counts[aa_position] += 1

                    # Separate forward and reverse ion types
                    if fragment_type.startswith(("a", "b", "c")):
                        forward_fragment_counts[aa_position] += 1
                    elif fragment_type.startswith(("x", "y", "z")):
                        reverse_fragment_counts[aa_position] += 1

        # Store the fragment counts for this PSM
        fragment_counts_by_psm[psm_id] = (
            all_fragment_counts,
            forward_fragment_counts,
            reverse_fragment_counts,
        )

    return fragment_counts_by_psm


def create_output_dataframe(
    results_df: pd.DataFrame,
    fragment_counts_by_psm: Dict[int, Tuple[List[int], List[int], List[int]]],
    mass_error_type: str = "ppm",
    mass_error_value: float = 50.0,
    use_mass_shift: bool = False,
) -> pd.DataFrame:
    """
    Create a DataFrame with ambiguity annotation for each peptide based on fragment counts.

    Args:
        results_df: DataFrame containing PSM data
        fragment_counts_by_psm: Dictionary mapping psm_id to a tuple of
            (all_fragment_counts, forward_fragment_counts, reverse_fragment_counts)
        mass_error_type: Type of mass error, either 'ppm' or 'Da'
        mass_error_value: Value of mass error threshold
        use_mass_shift: Whether to include mass shift annotation

    Returns:
        DataFrame with original results plus ambiguity_sequence column
    """
    logger = logging.getLogger(__name__)
    ambiguity_sequences, mass_shifts = [], []
    counts = {"valid": 0, "skipped": 0}

    # Create a copy of the results DataFrame to avoid modifying the original
    results_copy = results_df.copy()

    for _, psm_row in results_df.iterrows():
        psm_id = psm_row["psm_id"]

        if psm_id in fragment_counts_by_psm:
            counts["valid"] += 1
            _, forward_counts, reverse_counts = fragment_counts_by_psm[psm_id]

            # Calculate mass shift
            mass_shift = None
            if use_mass_shift:
                mass_shift = psm_row["expmass"] - psm_row["calcmass"]

                # Check if mass shift is within the expected range
                if mass_error_type == "ppm":
                    mass_error_ppm = (mass_shift / psm_row["calcmass"]) * 1e6
                    if abs(mass_error_ppm) <= mass_error_value:
                        # If within error range, don't consider it a true shift
                        mass_shift = None
                else:  # Da
                    if abs(mass_shift) <= mass_error_value:
                        # If within error range, don't consider it a true shift
                        mass_shift = None

            try:
                # Generate ambiguity annotation using peptacular
                ambiguity_sequence = pt.annotate_ambiguity(
                    psm_row["peptide"], forward_counts, reverse_counts, mass_shift
                )
                ambiguity_sequences.append(ambiguity_sequence)
                mass_shifts.append(mass_shift)
            except Exception as e:
                logger.error(f"Error annotating PSM {psm_id}: {str(e)}")
                ambiguity_sequences.append(None)
                mass_shifts.append(None)
        else:
            counts["skipped"] += 1
            logger.warning(f"PSM ID {psm_id} not found in fragment counts, skipping")
            ambiguity_sequences.append(None)
            mass_shifts.append(None)

    # Add the ambiguity sequences as a new column
    results_copy["ambiguity_sequence"] = ambiguity_sequences
    results_copy["mass_shift"] = mass_shifts

    logger.info(f"Processed {counts['valid']} PSMs with fragment counts")
    if counts["skipped"] > 0:
        logger.warning(f"Skipped {counts['skipped']} PSMs due to missing fragment data")

    return results_copy


def process_psm_data(
    results_df: pd.DataFrame,
    fragments_df: pd.DataFrame,
    mass_error_type: str = "ppm",
    mass_error_value: float = 50.0,
    use_mass_shift: bool = False,
) -> pd.DataFrame:
    """
    Process PSM data and add ambiguity annotations.

    Args:
        results_df: DataFrame containing PSM data
        fragments_df: DataFrame containing fragment ion data
        mass_error_type: Type of mass error, either 'ppm' or 'Da'
        mass_error_value: Value of mass error threshold
        use_mass_shift: Whether to include mass shift annotation

    Returns:
        DataFrame with original results plus ambiguity_sequence column
    """
    logger = logging.getLogger(__name__)

    # Count fragments per amino acid
    logger.info("Counting fragments per amino acid...")
    fragment_counts_by_psm = count_fragments_per_amino_acid(results_df, fragments_df)

    # Create output DataFrame with ambiguity annotations
    logger.info("Creating ambiguity annotations...")
    output_df = create_output_dataframe(
        results_df,
        fragment_counts_by_psm,
        mass_error_type=mass_error_type,
        mass_error_value=mass_error_value,
        use_mass_shift=use_mass_shift,
    )

    return output_df


def save_output(df: pd.DataFrame, output_path: str) -> None:
    """
    Save the output DataFrame to a file.

    Args:
        df: DataFrame to save
        output_path: Path to save the file to

    Raises:
        ValueError: If the file format is not supported
    """
    logger = logging.getLogger(__name__)

    # Create the directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")

    # Save based on file extension
    if output_path.endswith(".parquet"):
        logger.info(f"Saving output as parquet to {output_path}")
        df.to_parquet(output_path)
    elif output_path.endswith(".tsv"):
        logger.info(f"Saving output as TSV to {output_path}")
        df.to_csv(output_path, sep="\t", index=False)
    else:
        raise ValueError(
            "Unsupported output file format. Only .parquet and .tsv are supported."
        )

    logger.info(f"Output saved successfully: {output_path} ({len(df)} rows)")


def main():
    """Main function to run the application."""
    args = parse_arguments()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    try:
        logger.info("Starting SagePeptideAmbiguityAnnotator")

        # Read the input files
        results_df, fragments_df = read_input_files(args.results, args.fragments)

        # Process the data
        output_df = process_psm_data(
            results_df,
            fragments_df,
            mass_error_type=args.mass_error_type,
            mass_error_value=args.mass_error_value,
            use_mass_shift=args.mass_shift,
        )

        # Save the output
        save_output(output_df, args.output)

        logger.info("Processing completed successfully")
        return 0

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
