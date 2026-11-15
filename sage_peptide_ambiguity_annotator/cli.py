"""
Command line interface for SagePeptideAmbiguityAnnotator.

Three ways to name the input, one of which is required:

* ``--results`` with ``--fragments``: one pair of Sage output files.
* ``--folder``: one Sage output folder, whose ``results.sage.*`` and
  ``matched_fragments.sage.*`` files are found automatically.
* ``--data``: a directory of such folders, each processed in turn.
"""

import argparse
import logging
import os
import sys
from dataclasses import dataclass

from .utils import find_sage_files, process_psm_data, read_input_files, save_output


@dataclass
class Config:
    """Processing options, as resolved from the command line."""

    mass_error_type: str = "ppm"
    mass_error_value: float = 50.0
    use_mass_shift: bool = False
    verbose: bool = False
    output: str | None = None


class _HelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter
):
    """Shows each option's default, and leaves the epilog's line breaks alone."""


def setup_logging(verbose: bool = False) -> None:
    """
    Set up logging configuration.

    Args:
        verbose: Whether to show verbose (DEBUG) logs
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=log_level, format=log_format)


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Annotate Sage PSMs with peptide ambiguity based on fragment ion coverage.",
        formatter_class=_HelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s --results results.sage.parquet "
            "--fragments matched_fragments.sage.parquet --output annotated.sage.parquet\n"
            "  %(prog)s --folder /path/to/sage/output --output annotated.sage.tsv\n"
            "  %(prog)s --data /path/to/many/sage/outputs --output annotated.sage.tsv"
        ),
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-r",
        "--results",
        type=str,
        help="Path to the results.sage.parquet or results.sage.tsv file. Requires --fragments.",
    )
    group.add_argument(
        "-f",
        "--folder",
        type=str,
        help=(
            "Path to a folder holding one results.sage.* and one "
            "matched_fragments.sage.* file."
        ),
    )
    group.add_argument(
        "-d",
        "--data",
        type=str,
        help=(
            "Path to a directory of Sage output folders. Every immediate sub-folder is "
            "processed, and one output file is written inside each."
        ),
    )

    parser.add_argument(
        "-frag",
        "--fragments",
        type=str,
        help=(
            "Path to the matched_fragments.sage.parquet or matched_fragments.sage.tsv "
            "file. Required with --results."
        ),
    )

    # Mass error options. These set the threshold above which a precursor mass
    # difference is treated as a modification rather than as instrument mass error.
    parser.add_argument(
        "-type",
        "--mass_error_type",
        type=str,
        default="ppm",
        choices=["ppm", "Da"],
        help="Type of mass error (ppm or Da)",
    )
    parser.add_argument(
        "-value",
        "--mass_error_value",
        type=float,
        default=50.0,
        help="Value of mass error",
    )
    parser.add_argument(
        "-shift",
        "--mass_shift",
        action="store_true",
        help="Turn on mass shift annotation",
    )

    # Output options
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help=(
            "Path to the output file (.parquet or .tsv). With --data, its file name is "
            "used inside each sub-folder."
        ),
    )
    parser.add_argument(
        "-V",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args(argv)

    if args.results and not args.fragments:
        parser.error("--fragments is required when --results is specified.")
    if args.fragments and not args.results:
        parser.error("--results is required when --fragments is specified.")

    return args


def validate_paths(args: argparse.Namespace) -> None:
    """Validate that input paths exist."""
    if args.folder and not os.path.exists(args.folder):
        raise FileNotFoundError(f"Folder does not exist: {args.folder}")
    if args.results and not os.path.exists(args.results):
        raise FileNotFoundError(f"Results file does not exist: {args.results}")
    if args.fragments and not os.path.exists(args.fragments):
        raise FileNotFoundError(f"Fragments file does not exist: {args.fragments}")
    if args.data and not os.path.exists(args.data):
        raise FileNotFoundError(f"Data folder does not exist: {args.data}")


def get_folders_to_process(args: argparse.Namespace) -> list[tuple[str, str, str]]:
    """
    Determine which folders to process based on arguments.

    Returns:
        List of tuples (folder_path, results_path, fragments_path)

    Raises:
        ValueError: If a --data directory holds no sub-folder with Sage output in it
    """
    logger = logging.getLogger(__name__)

    if args.folder:
        results_path, fragments_path = find_sage_files(args.folder)
        return [(args.folder, results_path, fragments_path)]

    if args.results:
        return [(os.path.dirname(args.results), args.results, args.fragments)]

    sub_folders = sorted(
        os.path.join(args.data, name)
        for name in os.listdir(args.data)
        if os.path.isdir(os.path.join(args.data, name))
    )
    if not sub_folders:
        raise ValueError(f"No sub-folders found in data directory: {args.data}")

    folders_to_process = []
    for folder in sub_folders:
        try:
            results_path, fragments_path = find_sage_files(folder)
        except (FileNotFoundError, ValueError) as e:
            logger.warning(f"Skipping folder {folder}: {str(e)}")
            continue
        folders_to_process.append((folder, results_path, fragments_path))

    if not folders_to_process:
        raise ValueError(f"No valid folders found in data directory: {args.data}")

    return folders_to_process


def determine_output_path(args: argparse.Namespace, folder: str) -> str:
    """Determine the output path for a given folder."""
    if args.data:
        # One output file per sub-folder, named after the --output file name.
        return os.path.join(folder, os.path.basename(args.output))
    return args.output


def process_folder(
    folder: str,
    results_path: str,
    fragments_path: str,
    config: Config,
    output_path: str,
) -> bool:
    """
    Process a single folder.

    Returns:
        True if processing was successful, False otherwise
    """
    logger = logging.getLogger(__name__)

    try:
        logger.info(f"Processing folder: {folder}")

        results_df, fragments_df = read_input_files(results_path, fragments_path)

        output_df = process_psm_data(
            results_df,
            fragments_df,
            mass_error_type=config.mass_error_type,
            mass_error_value=config.mass_error_value,
            use_mass_shift=config.use_mass_shift,
        )

        if output_df.empty:
            logger.warning(f"No data to write after processing {folder}. Skipping.")
            return False

        # Saved before the check below on purpose: a run where every annotation failed
        # is reported as a failure, but the null-filled output is still written so the
        # failures can be inspected.
        save_output(output_df, output_path)

        if output_df["ambiguity_sequence"].notna().sum() == 0:
            logger.error(f"No PSM was annotated in {folder}")
            return False

        logger.info(f"Completed processing for folder: {folder}")
        return True

    except Exception as e:
        logger.error(f"Error processing folder {folder}: {str(e)}")
        if config.verbose:
            logger.exception("Full traceback:")
        return False


def main() -> int:
    """Main function to run the application."""
    args = parse_arguments()

    config = Config(
        mass_error_type=args.mass_error_type,
        mass_error_value=args.mass_error_value,
        use_mass_shift=args.mass_shift,
        verbose=args.verbose,
        output=args.output,
    )

    setup_logging(config.verbose)
    logger = logging.getLogger(__name__)

    logger.debug("Parsed arguments:")
    for arg, value in vars(args).items():
        logger.debug(f"  {arg}: {value}")

    try:
        logger.info("Starting SagePeptideAmbiguityAnnotator")

        validate_paths(args)
        folders_to_process = get_folders_to_process(args)

        successful_count = 0
        total_count = len(folders_to_process)

        for i, (folder, results_path, fragments_path) in enumerate(folders_to_process, 1):
            logger.info(f"Processing {i}/{total_count}: {folder}")
            output_path = determine_output_path(args, folder)
            if process_folder(folder, results_path, fragments_path, config, output_path):
                successful_count += 1

        logger.info(
            f"Processing completed: {successful_count}/{total_count} "
            "folders processed successfully"
        )
        return 0 if successful_count == total_count else 1

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=config.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
