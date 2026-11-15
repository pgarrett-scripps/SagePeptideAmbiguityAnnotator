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
annot-seq   (?PE)PT(?ID)[+100]E

# Observed mass shift is localized to: ID
```

```bash
sequence    PEPTIDE
b-ions      0111000
y-ions      0001111
mass-shift  100
annot-seq   {+100}(?PE)PTIDE

# since the forward and reverse ion series overlapped, the mass shift could not be localized and is added as a labile modification.
```

## Installation

Requires Python 3.12 or newer, the floor set by `peptacular>=5`.

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
              --output annotated_results.sage.parquet
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

Instead of naming both files, point `--folder` at one Sage output folder and the
`results.sage.*` and `matched_fragments.sage.*` files in it are found automatically. A
folder holding both a `.tsv` and a `.parquet` of the same kind is refused rather than
guessed at.

```bash
# One Sage output folder
sage-annotate --folder sage_results \
              --output annotated_results.sage.tsv
```

`--data` takes a directory of such folders and processes each one, writing the `--output`
file name inside every sub-folder. Sub-folders without Sage output are skipped with a
warning.

```bash
# A directory of Sage output folders
sage-annotate --data sage_data \
              --output annotated_results.sage.tsv
```

`--results`, `--folder` and `--data` are mutually exclusive, and exactly one is required.
`--fragments` is required with `--results`. Every flag has a short form:
`-r`, `-f`, `-d`, `-frag`, `-type`, `-value`, `-shift`, `-o`, `-V`.

### Streamlit Web Application

The web app needs Streamlit, which is not a dependency of the package:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Then open your browser at http://localhost:8501

### Streamlit Community Cloud
try me:
https://sage-peptide-ambiguity-annotator.streamlit.app/

## Python API

```python
from sage_peptide_ambiguity_annotator.utils import (
    read_input_files,
    process_psm_data,
    save_output,
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

The pipeline lives in `sage_peptide_ambiguity_annotator.utils` and the command line
interface in `sage_peptide_ambiguity_annotator.cli`. Importing these names from
`sage_peptide_ambiguity_annotator.main`, as releases up to 1.0.1 documented, still works.

### Finding the files in a folder

```python
from sage_peptide_ambiguity_annotator.utils import find_sage_files

results_path, fragments_path = find_sage_files("sage_results")
```

## Different Files / Single Peptides

```python
import peptacular as pt

# mass_shift is keyword-only in peptacular 5
ambiguity_sequence = pt.annotate_ambiguity(
    sequence='PEPTIDE',
    forward_coverage=[0,1,0,1,0,0,0],
    reverse_coverage=[0,0,0,1,1,1,0],
    mass_shift=100.0,
)
# '{+100.0}(?PE)PTI(?DE)'
```


## Input File Requirements

Both files may be `.parquet` or `.tsv`.

### Sage Results File

The Sage results file must have the following columns:
- `psm_id`: Unique identifier for each PSM
- `peptide`: The peptide sequence with modifications
- `expmass`: Experimental precursor mass
- `calcmass`: Calculated precursor mass

Optional:
- `stripped_peptide`: The peptide sequence without modifications. Derived from `peptide`
  with `peptacular.strip_mods` when the column is absent.

### Sage Matched Fragments File

The matched fragments file must have the following columns:
- `psm_id`: The PSM the fragment ion was matched to
- `fragment_type`: The fragment ion series (`a`, `b`, `c`, `x`, `y` or `z`)
- `fragment_ordinals`: The ordinal of the fragment ion within its series

## Output

The output file contains all columns from the input results file plus:
- `ambiguity_sequence`: Annotated peptide sequence with ambiguity information
- `mass_shift`: The observed mass shift, `expmass - calcmass`. Empty unless `--mass_shift`
  is passed, and empty for any PSM whose shift falls inside `--mass_error_value`, since
  such a difference is precursor mass error rather than a modification.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Dependencies

- pandas
- fastparquet
- peptacular >=5,<6
- streamlit (for web app, installed separately)