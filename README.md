# ThermADNet



```text


thermadnet-paper/
├── README.md
├── LICENSE
├── CITATION.cff
├── .gitignore
├── environment.yml            # or requirements.txt
├── Makefile                   # convenience: make reproduce, make figs, make eval
│
├── data/
│   ├── README.md              # where to download full data (Zenodo/DOI)
│   ├── sample/                # tiny samples used in unit tests / quick runs
│   └── external/              # (empty, ignored) placeholder for raw downloads
│
├── src/
│   ├── thermadnet/
│   │   ├── __init__.py
│   │   ├── io.py              # load/save helpers (csv, parquet, pickle)
│   │   ├── preprocessing.py   # missing-data handling, daily pivot, renaming
│   │   ├── features.py        # flags, aggregates, lookups
│   │   ├── models/            # AE/MLP/LSTM wrappers using autoencoder_helper
│   │   ├── evaluation.py      # F1/MCC/ROC, incident-centered metrics
│   │   └── plotting.py        # heatmaps, boxplots, ROC, threshold sweeps
│   └── scripts/
│       ├── prepare_data.py    # from raw → clean features + lookup tables
│       ├── train_models.py    # trains AE/LSTM/MLP, saves metrics/artifacts
│       ├── eval_thresholds.py # threshold sweeps, confusion matrices
│       └── make_figures.py    # regenerates all paper figures
│
├── notebooks/
│   ├── 00_exploration.ipynb   # (optional) compact EDA on sample data
│   ├── 10_prepare_data.ipynb  # thin wrapper calling src.scripts.prepare_data
│   ├── 20_train.ipynb         # thin wrapper calling src.scripts.train_models
│   └── 30_evaluate.ipynb      # thin wrapper calling src.scripts.eval_thresholds
│
├── paper/
│   ├── main.tex               # LaTeX (or your journal template)
│   ├── figures/               # auto-generated figures used in the paper
│   └── tables/                # CSV/tex tables auto-generated
│
├── results/
│   ├── metrics/               # csv/json of metrics, per experiment
│   ├── models/                # small weights or links/checkpoints (LFS)
│   └── logs/                  # training/runtime logs
│
└── tests/
    ├── test_io.py
    ├── test_preprocessing.py
    └── test_metrics.py
```