# Flight Departure Delay Prediction with PySpark

Predicting whether a US domestic flight will depart 15 or more minutes late, using PySpark MLlib on 7,079,081 flights from 2024.

- Course: DSAI4202, University of Doha for Science and Technology (UDST)
- Authors: Maryam Mahaboob and Amina Baziz

## Results at a glance

- **Data:** 7,079,081 flights in the raw 2024 data, 1,673,114 modelled after cleaning and sampling.
- **Best model:** Gradient Boosted Trees, with AUC-ROC 0.6975 and AUC-PR 0.3724 (baseline about 0.21, the share of delayed flights).
- **Threshold tuning:** at threshold 0.20, GBT catches 66.3% of delays, compared with 6.5% at the default 0.50.
- **Top drivers:** departure time, origin and destination airport, month, and carrier.

## Why this matters for airlines

- **On-time performance (OTP):** a main airline KPI. The US DOT counts a flight as late at 15 or more minutes, the same cutoff used here.
- **Operations control:** a pre-departure delay risk score gives time to position spare aircraft, call reserve crew, or protect tight turnarounds.
- **Hub connections:** at hubs, one late departure can break connections for transfer passengers, so flagging at-risk departures helps with rebooking and gate planning.

Full results, key insights, limitations and figures: [docs/analysis.md](docs/analysis.md)

## How to run

1. Download `flight_data_2024.csv` from [Kaggle](https://www.kaggle.com/datasets/hrishitpatil/flight-data-2024) (see [data/README.md](data/README.md)).
2. Upload it to Google Drive at `MyDrive/DSAI4202_Project/flight_data_2024.csv`, or change the path in the data loading cell.
3. Open the notebook in Google Colab and run the cells from top to bottom.

Requirements: see [requirements.txt](requirements.txt).
