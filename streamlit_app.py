"""
Streamlit app for SagePeptideAmbiguityAnnotator.

This app provides a web interface for the SagePeptideAmbiguityAnnotator tool,
allowing users to upload Sage output files, configure processing options,
and download the annotated results.
"""

import os
import tempfile
import streamlit as st
import logging

from sage_peptide_ambiguity_annotator.main import (
    read_input_files,
    process_psm_data,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="Sage Peptide Ambiguity Annotator",
    page_icon="🧬",
    layout="wide",
)


# Define function to create a download link
def create_download_link(df, filename, file_format="parquet"):
    """
    Create a download link for the processed data.

    Args:
        df: DataFrame to download
        filename: Name of the download file
        file_format: Format of the file (parquet or tsv)

    Returns:
        Download link HTML
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_format}") as temp:
        if file_format == "parquet":
            df.to_parquet(temp.name)
        else:  # tsv
            df.to_csv(temp.name, sep="\t", index=False)

        with open(temp.name, "rb") as f:
            data = f.read()

    os.unlink(temp.name)
    return data


# App title and description
st.title("Sage Peptide Ambiguity Annotator")
st.markdown(
    """
This app processes peptide spectrum matches (PSMs) from Sage search engine output 
and annotates peptides with ambiguity information based on fragment ion coverage.

Upload your Sage results file and matched fragments file to get started.
"""
)

# Sidebar with options
st.sidebar.header("Configuration")

# Mass error options
mass_error_type = st.sidebar.selectbox(
    "Mass Error Type",
    options=["ppm", "Da"],
    help="Type of mass error (parts per million or Daltons)",
)

mass_error_value = st.sidebar.number_input(
    "Mass Error Value",
    min_value=0.0,
    max_value=1000.0,
    value=50.0,
    help="Threshold value for mass error",
)

use_mass_shift = st.sidebar.checkbox(
    "Include Mass Shift Annotation",
    value=False,
    help="Whether to annotate mass shifts in peptides",
)

output_format = st.sidebar.selectbox(
    "Output Format", options=["parquet", "tsv"], help="File format for the output file"
)

# Main area with file upload
st.header("Upload Files", divider=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Sage Results File")
    results_file = st.file_uploader(
        "Upload results.sage.parquet or results.sage.tsv",
        type=["parquet", "tsv"],
        help="Sage search results file containing PSMs",
    )

with col2:
    st.subheader("Sage Fragments File")
    fragments_file = st.file_uploader(
        "Upload matched_fragments.sage.parquet or matched_fragments.sage.tsv",
        type=["parquet", "tsv"],
        help="Sage matched fragment ions file",
    )

# Process button
if st.button(
    "Process Files", disabled=(results_file is None or fragments_file is None)
):
    if results_file is None or fragments_file is None:
        st.error("Please upload both required files.")
    else:
        try:
            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Save uploaded files to temp location
            status_text.text("Saving uploaded files...")
            progress_bar.progress(10)

            results_suffix = (
                ".parquet" if results_file.name.endswith(".parquet") else ".tsv"
            )
            fragments_suffix = (
                ".parquet" if fragments_file.name.endswith(".parquet") else ".tsv"
            )

            with tempfile.NamedTemporaryFile(
                delete=False, suffix=results_suffix
            ) as temp_results:
                temp_results.write(results_file.getvalue())
                results_path = temp_results.name

            with tempfile.NamedTemporaryFile(
                delete=False, suffix=fragments_suffix
            ) as temp_fragments:
                temp_fragments.write(fragments_file.getvalue())
                fragments_path = temp_fragments.name

            # Read input files
            status_text.text("Reading input files...")
            progress_bar.progress(20)
            results_df, fragments_df = read_input_files(results_path, fragments_path)

            # Process data
            status_text.text(f"Processing {len(results_df)} PSMs...")
            progress_bar.progress(40)
            output_df = process_psm_data(
                results_df,
                fragments_df,
                mass_error_type=mass_error_type,
                mass_error_value=mass_error_value,
                use_mass_shift=use_mass_shift,
            )

            # Create output
            progress_bar.progress(80)
            status_text.text("Preparing download...")

            # Generate output filename
            output_filename = f"annotated_results.sage.{output_format}"

            # Create download button
            download_data = create_download_link(
                output_df, output_filename, output_format
            )
            progress_bar.progress(100)
            status_text.text("Processing complete!")

            # Clean up temp files
            os.unlink(results_path)
            os.unlink(fragments_path)

            # Show download button
            st.success(f"Successfully processed {len(output_df)} PSMs!")
            st.download_button(
                label=f"Download Annotated Results ({output_format})",
                data=download_data,
                file_name=output_filename,
                mime="application/octet-stream",
            )

            # Display sample of results
            st.subheader("Sample Results (First 5 rows)")
            st.dataframe(output_df.head(5))

            # Display stats
            st.subheader("Statistics")
            st.write(f"Total PSMs processed: {len(output_df)}")
            st.write(
                f"PSMs with ambiguity annotation: {output_df['ambiguity_sequence'].notna().sum()}"
            )

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logger.exception("Error processing files")

# Add explanatory section
st.header("How It Works")
st.markdown(
    """
The SagePeptideAmbiguityAnnotator processes your PSM data in the following steps:

1. **Read Input Files**: Parses the Sage results and fragment files.
2. **Map Fragments**: Maps each fragment ion to its corresponding amino acid position.
3. **Count Fragments**: Counts the number of fragment ions matching each amino acid.
4. **Annotate Ambiguity**: Creates an ambiguity annotation based on fragment coverage.
5. **Output Results**: Adds the ambiguity_sequence column to the original results.

The ambiguity annotation indicates which parts of the peptide sequence have strong 
fragment ion evidence and which parts are less certain.
"""
)

# Footer
st.sidebar.markdown("---")
st.sidebar.info(
    "SagePeptideAmbiguityAnnotator - Version 0.1.0\n\n"
    "For more information, visit the [GitHub repository](https://github.com/yourusername/SagePeptideAmbiguityAnnotator)."
)
