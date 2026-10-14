# Dataset setup (ACNE04)

The dataset is NOT included in this repository (size + licence). To set it up:

1. Obtain ACNE04 from the location given by its authors. Wu et al. (ICCV 2019) release it via
   https://github.com/xpwu95/LDL (the link cited by Zhang et al., 2024). **Read the licence/usage terms
   on that page** and keep to them (typically research/educational use, cite the original paper).
2. Unzip so the images are somewhere under `data/ACNE04/` (any sub-folder layout is fine).
3. Inspect it (nothing is assumed about the layout):
   `python scripts/inspect_dataset.py --data-root data/ACNE04`
4. Prepare labels + splits:
   `python scripts/prepare_dataset.py --data-root data/ACNE04 --label-source filename`
   (or `--label-source csv --labels-csv labels.csv` with columns `filename,label`).

Images/labels are never uploaded anywhere. Do not commit the dataset (`.gitignore` already excludes it).
