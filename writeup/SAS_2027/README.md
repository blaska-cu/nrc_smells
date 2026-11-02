# SAS 2027 reproduction notebook

Open **[reproduce_paper.ipynb](reproduce_paper.ipynb)** and run **Restart Kernel and
Run All**. It reproduces the four tables and four figures in *Unsupervised Model
Transfer Between Temperature-Modulated MOX Gas Sensors*.

All analysis code is inside the notebook. Its only measurement inputs are:

```text
bme_690_all_nuts.csv
bme_690_all_spices.csv
bme_690_all_herbs.csv
```

In this repository they are in `smell_project/BME690_data/2026collection`. The
notebook finds that directory when launched from the repository root, this
directory, or an intervening directory. When distributing the notebook separately,
set `DATA_DIR` in the first code cell to the folder containing those three files.
The CSVs are not duplicated here.

Use Python 3.10.11 and install the publication dependencies in your chosen environment:

```shell
python -m pip install -r requirements.txt
```

Open the notebook in a Jupyter-compatible editor and select that Python
environment as the kernel. For a browser interface, JupyterLab can be installed
separately with `python -m pip install jupyterlab`.

The delivered notebook includes executed outputs. Rerunning it recomputes every
result from the measurement files, including the 30-trial adaptation coverage
sweep. Tables and figures are displayed inline; no figures, CSVs, or LaTeX files
are exported. There are no imports from the original experiment modules and no
dependency on the original `writeup/results` directory. The heater schedules are
included as configuration data checked against the available raw recording
headers, so the `.bmerawdata` files are unnecessary.

| Notebook section | Paper output |
| --- | --- |
| Dataset and method summaries | Tables I and II |
| Heater profiles | Figure 1 |
| Supervised LDA projections | Figure 2 |
| Paired-sensor fingerprints | Figure 3 |
| Transfer ablation | Table III, including source training scores in its caption |
| Recall for selected substances | Table IV |
| Reduced adaptation coverage | Figure 4 |

The notebook displays dataset hashes and environment versions, and checks input
row, cycle, and class counts. Keep the pinned numerical-library versions for exact
numerical reproduction: newer versions can slightly change LDA decisions near
class boundaries.

CORAL's fixed ridge is applied in **source-standardized resistance coordinates**.
Vapor pressure is never target-aligned. The coverage bands are the 10th–90th
percentiles over class subsets after averaging transfer directions within each
trial; they are not confidence intervals. The notebook explains these details
next to the relevant calculations.

Plot fonts and label placement favor readability in the notebook; the underlying
data, projections, selected classes, and adaptation protocols reproduce the paper.
The existing paper source and its figures are not updated by running the notebook.
