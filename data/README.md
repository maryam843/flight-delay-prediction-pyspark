# Data

The raw data is not included in this repository. Download it from Kaggle:

https://www.kaggle.com/datasets/hrishitpatil/flight-data-2024

- File: `flight_data_2024.csv` (the full file, not the sample file)
- Size: 7,079,081 rows x 35 columns, about 1.3 GB
- Scope: US domestic flights in 2024

## Columns used

The notebook reads all 35 columns and then keeps these 11:

| Column | Use |
|---|---|
| `month` | feature (sine/cosine encoded) |
| `day_of_week` | feature (sine/cosine encoded) |
| `day_of_month` | feature (sine/cosine encoded) |
| `crs_dep_time` | feature: scheduled departure time (sine/cosine encoded) |
| `op_unique_carrier` | feature: carrier code (StringIndexer) |
| `origin` | feature: origin airport (StringIndexer) |
| `dest` | feature: destination airport (StringIndexer) |
| `distance` | feature |
| `crs_elapsed_time` | used in EDA only, dropped as a feature (0.98 correlation with distance) |
| `dep_delay` | target source: label is 1 when `dep_delay` >= 15 minutes |
| `cancelled` | filter: only rows with `cancelled == 0` are kept |

## Where to put it in Colab

The notebook mounts Google Drive and reads the file from this path:

```
/content/drive/MyDrive/DSAI4202_Project/flight_data_2024.csv
```

Create a `DSAI4202_Project` folder in your Google Drive and upload the CSV there,
or change the path in the data loading cell. The notebook also saves its plots
to the same folder.

CSV and Parquet files are listed in `.gitignore`, so a local copy placed in this
folder will not be committed.
