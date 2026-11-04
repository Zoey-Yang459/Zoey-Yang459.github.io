# SYDE 572 — Assignment 3 (PCA and ISOMAP)

This folder contains a complete GitHub Pages website for Assignment 3. It is designed to live **inside the existing GitHub Pages repository**, without replacing the Assignment 1 or Assignment 2 webpages.

## Publish at a separate URL

Copy this entire directory to the root of your GitHub Pages repository and name the directory **`assignment3`**. Keep the nested `plots/` and `assets/formulas/` directories intact.

For the repository `Zoey-Yang459.github.io`, the resulting URL is:

`https://zoey-yang459.github.io/assignment3/`

Do **not** overwrite the root `index.html` (Assignment 1) or the `assignment2/index.html` page.

### Recommended folder structure

```
Zoey-Yang459.github.io/
├── index.html                   # existing Assignment 1
├── assignment2/                # existing Assignment 2
│   └── index.html
└── assignment3/                # this directory
    ├── index.html
    ├── Assignments3-Part1.pdf       # original checked handwritten calculations
    ├── SYDE572_Assignment3_Report.pdf
    ├── README.md
    ├── requirements.txt
    ├── Part_2.py
    ├── Part_3.py
    ├── Part_5.py
    ├── plots/                   # 9 figures
    └── assets/
        ├── formulas/           # pre-rendered vector math equations
        └── handwritten/        # full-size images of each PDF page
```

### How to upload

**Using GitHub Desktop (simplest for nested folders):** Clone/open the existing GitHub Pages repository on your Mac, copy the whole `assignment3` folder into the repository root, commit, and push.

**Using GitHub in the browser:** Make sure the `assignment3` directory is created at the repository root, then upload `index.html`, the `.py` files, the PDF, **and both nested directories** to their respective paths. GitHub stores files rather than empty folders; uploading just the HTML without `plots/` and `assets/formulas/` will lead to broken images or equations.

## Files and methodology

- `index.html`: Full report + figures + entire Python source, always visible and printable.
- `SYDE572_Assignment3_Report.pdf`: Updated printable report including all 3 handwritten pages before the Python code.
- `Assignments3-Part1.pdf`: Your original, mathematically verified handwritten Part 1 solutions.
- `assets/handwritten/`: Full-page previews from the PDF so every page is visible and printable in the webpage.
- `Part_2.py`: One-, two-, and three-feature Iris visualizations.
- `Part_3.py`: PCA axes, 1D/2D/3D projections, and Part 4 data-compression calculations.
- `Part_5.py`: ISOMAP (`n_neighbors=3`), PCA comparison, and silhouette scores.
- `plots/`: All nine figures supplied from the completed assignment runs.
- `assets/formulas/`: Local SVG mathematical equations; the website does not need MathJax or an internet connection to render its formulas.

PCA is applied to mean-centered **original centimetre measurements**, without z-score standardization. `Isomap(n_neighbors=3, n_components=2)` yields two connected components in the underlying nearest-neighbor graph; scikit-learn bridges them. Small version-dependent differences in the ISOMAP score can occur; the reported score (0.559967) comes from the student's run.

**Original user-uploaded Python scripts:** The `Part_2.py`, `Part_3.py`, and `Part_5.py` files, and the code appendix in `index.html`, are exact copies of the student's uploaded source files. The mathematical methods were not changed.

## Re-run the scripts

In a Python environment with the required packages installed:

```bash
python -m pip install -r requirements.txt
python Part_2.py
python Part_3.py
python Part_5.py
```

Running the scripts regenerates figures under `plots/`, and plot sizes/ISOMAP positions may vary slightly by local environment or software version.

## Submission reminder

Check the published GitHub Pages URL, then submit your PDF and/or website link according to the assignment's required submission instructions.
