"""
SagePeptideAmbiguityAnnotator main module.

This module processes peptide spectrum matches (PSMs) from Sage search engine output
and annotates peptides with ambiguity information based on fragment ion coverage.
"""

import sys
import logging
import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional
from pathlib import Path

from .utils import read_input_files, process_psm_data, save_output, find_sage_files


@dataclass
class Config:
    """Configuration class for the application."""
    mass_error_type: str = "ppm"
    mass_error_value: float = 50.0
    use_mass_shift: bool = False
    verbose: bool = False
    output: Optional[str] = None


def setup_logging(verbose: bool = False) -> None:
    """
    Set up logging configuration.

    Args:
        verbose: Whether to show verbose (DEBUG) logs
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=log_level, format=log_format)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Annotate Sage PSMs with peptide ambiguity based on fragment ion coverage.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s --folder /path/to/sage/output --output results.tsv\n"
            "  %(prog)s --results results.sage.parquet --fragments matched_fragments.sage.parquet --output results.tsv"
        ),
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-f",
        "--folder",
        type=str,
        help="Path to folder containing results.sage.* and matched_fragments.sage.* files.",
    )
    group.add_argument(
        "-r",
        "--results",
        type=str,
        help="Path to results.sage.parquet or results.sage.tsv file.",
    )
    group.add_argument(
        "-d",
        "--data",
        type=str,
        help="Path to data folder containing many sage output files.",
    )

    parser.add_argument(
        "-frag",
        "--fragments",
        type=str,
        help="Path to matched_fragments.sage.parquet or matched_fragments.sage.tsv file (required if --results is used).",
    )

    parser.add_argument(
        "-type",
        "--mass_error_type",
        type=str,
        default="ppm",
        choices=["ppm", "Da"],
        help="Type of mass error to use for fragment matching.",
    )
    parser.add_argument(
        "-value",
        "--mass_error_value",
        type=float,
        default=50.0,
        help="Mass error tolerance value.",
    )
    parser.add_argument(
        "-shift",
        "--mass_shift",
        action="store_true",
        help="Enable mass shift annotation.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Path to output file (.parquet or .tsv).",
    )
    parser.add_argument(
        "-V",
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging.",
    )

    args = parser.parse_args()

    # Validate arguments
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


def get_folders_to_process(args: argparse.Namespace) -> List[Tuple[str, str, str]]:
    """
    Determine which folders to process based on arguments.
    
    Returns:
        List of tuples (folder_path, results_path, fragments_path)
    """
    logger = logging.getLogger(__name__)
    
    if args.folder:
        results_path, fragments_path = find_sage_files(args.folder)
        return [(args.folder, results_path, fragments_path)]
    
    elif args.results:
        results_path = args.results
        fragments_path = args.fragments
        return [(os.path.dirname(results_path), results_path, fragments_path)]
    
    elif args.data:
        sub_folders = [
            os.path.join(args.data, d)
            for d in os.listdir(args.data)
            if os.path.isdir(os.path.join(args.data, d))
        ]
        if not sub_folders:
            raise ValueError(f"No sub-folders found in data directory: {args.data}")
        
        folders_to_process = []
        for folder in sub_folders:
            try:
                results_path, fragments_path = find_sage_files(folder)
                folders_to_process.append((folder, results_path, fragments_path))
            except Exception as e:
                logger.warning(f"Skipping folder {folder}: {str(e)}")
        
        if not folders_to_process:
            raise ValueError(f"No valid folders found in data directory: {args.data}")
        
        return folders_to_process
    
    else:
        raise ValueError("No valid input method specified")


def determine_output_path(args: argparse.Namespace, folder: str, results_path: str) -> str:
    """Determine the output path for a given folder."""
    if args.output:
        if args.data:
            # For data mode, create output files in each subfolder
            output_name = os.path.basename(args.output)
            return os.path.join(folder, output_name)
        else:
            return args.output
    else:
        # Generate default output name
        results_file_type = os.path.splitext(os.path.basename(results_path))[1]
        output_name = f"annotated_results{results_file_type}"
        return os.path.join(folder, output_name)


def process_folder(folder: str, results_path: str, fragments_path: str, 
                  config: Config, output_path: str) -> bool:
    """
    Process a single folder.
    
    Returns:
        True if processing was successful, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Processing folder: {folder}")
        
        # Read the input files
        results_df, fragments_df = read_input_files(results_path, fragments_path)

        # Process the data
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

        # Save the output
        save_output(output_df, output_path)
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
    
    
    # Create configuration
    config = Config(
        mass_error_type=args.mass_error_type,
        mass_error_value=args.mass_error_value,
        use_mass_shift=args.mass_shift,
        verbose=args.verbose,
        output=args.output
    )

    # Setup logging
    setup_logging(config.verbose)
    logger = logging.getLogger(__name__)

    # print all args pretty
    logger.info("Parsed arguments:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")

    try:
        logger.info("Starting SagePeptideAmbiguityAnnotator")

        # Validate input paths
        validate_paths(args)

        # Get folders to process
        folders_to_process = get_folders_to_process(args)
        
        # Process each folder
        successful_count = 0
        total_count = len(folders_to_process)
        
        for i, (folder, results_path, fragments_path) in enumerate(folders_to_process, 1):
            logger.info(f"Processing {i}/{total_count}: {folder}")
            
            # Determine output path
            output_path = determine_output_path(args, folder, results_path)
            
            # Process the folder
            if process_folder(folder, results_path, fragments_path, config, output_path):
                successful_count += 1

        logger.info(f"Processing completed: {successful_count}/{total_count} folders processed successfully")
        return 0 if successful_count > 0 else 1

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        if config.verbose:
            logger.exception("Full traceback:")
        return 1


if __name__ == "__main__":
    sys.exit(main())
