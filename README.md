pydrift
==============================

How do we measure the degradation of a machine learning process? Why does the performance of our predictive models decrease? Maybe it is that a data source has changed (one or more variables) or maybe what changes is the relationship of these variables with the target we want to predict. `pydrift` tries to facilitate this task to the data scientist, performing this kind of checks and somehow measuring that degradation.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make test`
    ├── README.md          <- The top-level README for developers using this project.
    ├── datasets
    ├── docs               <- Sphinx project; see sphinx-doc.org for details
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-scp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── pydrift            <- Source code for use in this project.
    │   ├── __init__.py    <- Makes pydrift a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── pyproject.toml     <- poetry file


--------
