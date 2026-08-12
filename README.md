# Antwerp eclipse viewshed

This project tests whether buildings and vegetation will obstruct the partial solar eclipse from five viewing locations in Antwerp on 12 August 2026.

The analysis combines the eclipse path with one-metre elevation data. For each viewing location, it traces the western skyline and measures when the eclipsed Sun remains above it.

## Workflow

Run the notebooks in order:

1. `01_sun_geometry.ipynb` calculates the eclipse contacts and the Sun's path.
2. `02_acquire_elevation.ipynb` downloads and mosaics the digital surface and terrain models.
3. `03_surface_model.ipynb` classifies water, buildings, vegetation and open ground.
4. `04_horizon_analysis.ipynb` constructs each skyline and calculates visibility.
5. `05_skylines.ipynb` produces the two skyline figures and an independent summary table.
6. `06_site_map.ipynb` produces the viewing-location map.

Each notebook reads the saved outputs of the preceding stages. Notebook outputs are left clear in versioned files; rerunning a notebook writes its figures and data products to disk.

## Viewing locations

Coordinates and eye-height assumptions are stored in `data/external/observers.csv`. The five locations are:

- MAS panoramic platform
- Droogdokkenpark
- Scheldekaaien Zuid
- Nieuw Zuid
- Wandelterras, Het Steen

The MAS rooftop coordinate was located manually and checked against Google Maps and the digital surface model. The Wandelterras uses a five-metre platform height plus a 1.6-metre eye height.

## Data

- Solar and eclipse geometry calculated in `src/eclipse_viewshed/`
- Digital surface and terrain models from Digitaal Vlaanderen's elevation service
- Building footprints from the GRB Web Feature Service
- Water classification from ESA WorldCover
- Road and water features used for the site map

The elevation surface dates from 2013–2015. Current 3D building information supplements newer development in the tested area on Linkeroever; later vegetation changes are not represented. Source details, licences and the Solar Position Algorithm citation are recorded in `references/data_sources.md`.

A complete run downloads about 1.3 GB of source data and creates about 275 MB of intermediate rasters. Internet access is required for notebooks 02, 03 and 06. The downloaded and intermediate files are reproducible and are not stored in Git.

## Outputs

- Downloaded source data: `data/raw/`
- Intermediate rasters and vectors: `data/interim/`
- Horizon profiles and summary tables: `data/processed/`
- Analysis figures: `reports/figures/`
- Finished webpage: `reports/factsheet/factsheet.html`
- Website export: generated outside the repository from the finished webpage

## Environment and checks

Create the environment and install this package in editable mode:

```text
conda env create -f environment.yml
conda activate eclipse-viewshed
python -m pip install --no-deps -e .
python check_environment.py
jupyter lab
```

Open the notebooks from the repository root and run notebooks 01 through 06 in order. Their paths are resolved from the repository marker, so Jupyter does not need to start in a particular subdirectory.

The automated checks are:

```text
pytest -q
python tools/verify_factsheet.py
```

The first command tests the analysis modules. The second compares the factsheet's times, table and figures with the processed results.

GitHub Actions runs the unit tests for each push and pull request. It does not download the full geospatial dataset or execute the notebooks.

## Licence

The project code is released under the MIT License. The source datasets and reference photograph retain their own terms, listed in `references/data_sources.md`.
