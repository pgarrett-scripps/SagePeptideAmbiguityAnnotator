# SagePeptideAmbiguityAnnotator

A tool for annotating peptide ambiguity in Sage search engine results based on fragment ion coverage.

## Description

The SagePeptideAmbiguityAnnotator processes peptide spectrum matches (PSMs) from Sage search engine output and annotates peptides with ambiguity information based on fragment ion coverage. It helps identify which parts of a peptide sequence have strong evidence from fragment ions and which parts are less certain. For open searches it can also place the observed mass shift as an internal modification, or labile modification if complete fragment ion coverage is observed.

## Examples (simple)

```bash
sequence    PEPTIDE
b-ions      0110000
y-ions      0000110
annot-seq   (?PE)PTI(?DE)
```

```bash
sequence    PEPTIDE
b-ions      0111000
y-ions      0000000
annot-seq   (?PE)PT(?IDE)
```

## Examples (open search)


```bash
sequence    PEPTIDE
b-ions      0111000
y-ions      0000001
mass-shift  100
annot-seq   (?PE)PT(?ID)[100]E

# Observed mass shift is localized to: ID
```

```bash
sequence    PEPTIDE
b-ions      0111000
y-ions      0001111
mass-shift  100
annot-seq   {100}(?PE)PTIDE

# since the forward and reverse ion series overlapped, the mass shift could not be localized and is added as a labile modification.
```

## Installation

### From PyPI

```bash
pip install sage-peptide-ambiguity-annotator
```

### From Source

```bash
git clone https://github.com/pgarrett-scripps/SagePeptideAmbiguityAnnotator.git
cd SagePeptideAmbiguityAnnotator
pip install -e .
```

## Usage

### Command Line Interface

```bash
# Normal Search (Without Mass Shifts)
sage-annotate --results results.sage.parquet \
              --fragments matched_fragments.sage.parquet \
              --output annotated_results.sage.parquet \
```

```bash
# Open Search (With Mass Shifts)
sage-annotate --results results.sage.parquet \
              --fragments matched_fragments.sage.parquet \
              --output annotated_results.sage.parquet \
              --mass_error_type ppm \
              --mass_error_value 50.0 \
              --mass_shift
```

### Streamlit Web Application

```bash
streamlit run streamlit_app.py
```

Then open your browser at http://localhost:8501

### Streamlit Community Cloud
try me:
https://sage-peptide-ambiguity-annotator.streamlit.app/

## Python API

```python
from sage_peptide_ambiguity_annotator.main import (
    read_input_files, 
    process_psm_data, 
    save_output
)

# Read input files
results_df, fragments_df = read_input_files(
    "results.sage.parquet", 
    "matched_fragments.sage.parquet"
)

# Process the data
output_df = process_psm_data(
    results_df, 
    fragments_df,
    mass_error_type="ppm",
    mass_error_value=50.0,
    use_mass_shift=True
)

# Save the output
save_output(output_df, "annotated_results.sage.parquet")
```

## Different Files / Single Peptides

```python
import peptacular as pt

ambiguity_sequence = pt.annotate_ambiguity(
    sequence='PEPTIDE', 
    forward_coverage=[0,1,0,1,0,0,0], 
    reverse_coverage=[0,0,0,1,1,1,0], 
    mass_shift=100.0
)
```


## Input File Requirements

### Sage Results File

The Sage results file must have the following columns:
- `psm_id`: Unique identifier for each PSM
- `peptide`: The peptide sequence with modifications
- `stripped_peptide`: The peptide sequence without modifications
- `expmass`: Experimental mass
- `calcmass`: Calculated mass

## Output

The output file contains all columns from the input results file plus:
- `ambiguity_sequence`: Annotated peptide sequence with ambiguity information
- `mass_shift`: The observed mass shift between the experimental and observed precursor masses. (Only applicable with open search)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Dependencies

- pandas
- fastparquet
- peptacular
- streamlit (for web app)