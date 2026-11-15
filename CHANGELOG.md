# Changelog

## Unreleased

### Added

- Folder and data folder support. `--folder` takes one Sage output folder and finds its
  `results.sage.*` and `matched_fragments.sage.*` files; `--data` takes a directory of
  such folders and processes each one, writing the `--output` file name inside every
  sub-folder. `--results`, `--folder` and `--data` are mutually exclusive and exactly one
  is required.
- Short forms for every command line flag: `-r`, `-f`, `-d`, `-frag`, `-type`, `-value`,
  `-shift`, `-o`, `-V`.
- `find_sage_files(folder_path)` in `sage_peptide_ambiguity_annotator.utils`, which
  returns the `(results_path, fragments_path)` pair for a folder and refuses a folder
  holding both a `.tsv` and a `.parquet` of the same kind rather than guessing.
- A test suite: 71 tests over a committed five-PSM example, with the annotation output
  recorded on peptacular 2.5.0 and on 5.x so the port is pinned in both directions.

### Changed

- **Annotation output: positive mass values now carry an explicit `+`.** peptacular 5
  writes `[+100.0]` where 2.5.0 wrote `[100.0]`, `{+100.0}` where it wrote `{100.0}`, and
  `[+57.0215]` where it wrote `[57.0215]`, for both the placed mass shift and
  modifications already on the input peptide. Ambiguity intervals, the localization
  decision, the labile fallback and the `mass_shift` column are unchanged.
- Requires `peptacular>=5,<6`, and therefore Python 3.12 or newer. Release 1.0.1 pinned
  `peptacular==2.5.0`; on peptacular 5 that code raised `TypeError` inside a per-PSM
  `except Exception` and left every `ambiguity_sequence` null while still exiting 0.
- The pipeline moved from `sage_peptide_ambiguity_annotator.main` to
  `sage_peptide_ambiguity_annotator.utils`, and the command line interface to
  `sage_peptide_ambiguity_annotator.cli`. `main` re-exports both, so existing imports and
  the `sage-annotate` entry point keep working.
- CI runs `ruff check` and `pytest` on Python 3.12 and 3.13. It previously installed
  pytest without running it.

### Fixed

- `sage-annotate` counted PSMs that had fragment counts rather than PSMs that were
  annotated, and a PSM whose annotation raised was logged and written as a null row
  without failing the run: ten of ten null rows still reported "Processing completed
  successfully" and exited 0. It now counts annotation failures, logs one summary line
  (`Annotated N of M PSMs ...`), and exits non-zero when no PSM was annotated. The output
  file is still written in that case, so the null rows can be inspected. (#3)
- Two tests compared paths with a hard-coded `/`, so they failed on Windows while passing
  on Linux and in CI. Reported by bfs2021 while contributing #6.
- Several README claims that did not match the code: `stripped_peptide` is derived rather
  than required, the matched-fragments columns are documented, `mass_shift` is
  `expmass - calcmass` and is empty without `--mass_shift` or inside the mass error
  threshold, and the Python floor is stated.
- The Streamlit app's footer reported version 0.1.0 and linked to a placeholder
  `github.com/yourusername/...` URL.
