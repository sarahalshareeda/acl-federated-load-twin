# data/

This folder ships empty. Building Data Genome 2 is a public dataset and is not
re-hosted here.

`bdg2_federated_split.ipynb` reads exactly three files from it:

```
data/
  meters/cleaned/electricity_cleaned.csv     177 MB
  weather/weather.csv                         19 MB
  metadata/metadata.csv                      0.3 MB
```

About 196 MB in total. The rest of BDG2, including the other meter types, the
raw meters, and the 897 MB `kaggle.csv`, is never read and does not need
downloading.

## Where to get them

- Zenodo, with a DOI: https://doi.org/10.5281/zenodo.3887306
- GitHub: https://github.com/buds-lab/building-data-genome-project-2

Download those three, recreate the folder structure above, and drop them in.
The split notebook checks each path before it starts and exits naming the
missing file, so a wrong path fails immediately rather than halfway through.

## Configuration

Two settings at the top of the split notebook decide which meter file is read:

```python
TARGET_METER = "electricity"
USE_CLEANED  = True
```

`USE_CLEANED = False` reads `data/meters/raw/electricity.csv` instead, which is
a similar size.

## You may not need this at all

`data/` is only required if you want to rebuild the client parquets from the
raw dataset. The 60 parquets are already in `bdg2_federated_out/clients/`, so
training, calibration, the figures and the twin all run without downloading
anything.
