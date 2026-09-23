# Flight Departure Delay Prediction with PySpark

Predicting whether a US domestic flight will depart 15 or more minutes late, using PySpark MLlib on 7,079,081 flights from 2024.

- Course: DSAI4202, University of Doha for Science and Technology (UDST)
- Authors: Maryam Mahaboob and Amina Baziz

## Results at a glance

- **Data:** 7,079,081 flights in the raw 2024 data, 1,673,114 modelled after cleaning and sampling.
- **Best model:** Gradient Boosted Trees, with AUC-ROC 0.6975 and AUC-PR 0.3724 (baseline about 0.21, the share of delayed flights).
- **Threshold tuning:** at threshold 0.20, GBT catches 66.3% of delays, compared with 6.5% at the default 0.50.
- **Top drivers:** departure time, origin and destination airport, month, and carrier.

![GBT feature importance](figures/gbt_feature_importance.png)

## Why this matters for airlines

- **On-time performance (OTP):** a main airline KPI. The US DOT counts a flight as late at 15 or more minutes, the same cutoff used here.
- **Operations control:** a pre-departure delay risk score gives time to position spare aircraft, call reserve crew, or protect tight turnarounds.
- **Hub connections:** at hubs, one late departure can break connections for transfer passengers, so flagging at-risk departures helps with rebooking and gate planning.

## Approach

- **Pre-departure features only:** schedule, carrier, airports and distance. `dep_delay` is used only to build the label.
- **Cyclical encoding:** month, day of week, day of month and departure time as sine/cosine pairs.
- **StringIndexer + Pipelines:** carrier and airports indexed inside MLlib Pipelines fitted on training data only.
- **Class weights:** 20.6% of flights are delayed (3.86 : 1), so LR and RF use a weight of 3.86 on delayed flights.
- **CrossValidator + threshold tuning:** 3-fold CV over GBT depth and iterations, then the decision threshold is tuned without retraining.

## Results

Test set of 334,277 flights. F1 is weighted across both classes.

| Model | Accuracy | Weighted F1 | AUC-ROC | AUC-PR |
|---|---|---|---|---|
| Logistic Regression | 0.5814 | 0.6215 | 0.6519 | 0.3058 |
| Random Forest | 0.6062 | 0.6440 | 0.6674 | 0.3368 |
| GBT (untuned) | 0.7963 | 0.7270 | 0.6975 | 0.3724 |
| GBT (tuned, CV) | 0.7963 | 0.7270 | 0.6975 | 0.3724 |

GBT threshold tuning (delayed class):

| Threshold | Accuracy | Precision | Recall | F1 (delayed) |
|---|---|---|---|---|
| 0.10 | 0.2957 | 0.2218 | 0.9689 | 0.3610 |
| 0.15 | 0.4846 | 0.2637 | 0.8424 | 0.4016 |
| **0.20** | **0.6325** | **0.3134** | **0.6631** | **0.4256** |
| 0.25 | 0.7138 | 0.3590 | 0.5016 | 0.4185 |
| 0.30 | 0.7585 | 0.4017 | 0.3598 | 0.3796 |
| 0.35 | 0.7811 | 0.4434 | 0.2589 | 0.3269 |
| 0.40 | 0.7933 | 0.4897 | 0.1567 | 0.2374 |
| 0.45 | 0.7963 | 0.5226 | 0.0902 | 0.1539 |
| 0.50 | 0.7963 | 0.5323 | 0.0650 | 0.1159 |

![Threshold tuning](figures/threshold_tuning.png)

The red line in the plot marks 0.35, which the plot code labels "Suggested threshold". The threshold chosen in the analysis is 0.20, where delayed-class F1 is highest.

## Key insights

- **Departure time is the top feature** (GBT importance about 0.23). Delays build up over the day as earlier late arrivals push back later departures.
- **July and Friday peaks:** July has the highest average delay (about 22 minutes) and Friday the highest by day of week (about 14 to 15 minutes).
- **Carriers differ:** F9 (Frontier) has the highest delay rate at about 28%. HA (Hawaiian) and YX have the lowest.
- **Hub airports:** among the 10 busiest airports, DFW (about 27%) and CLT (about 26%) have the highest delay rates.
- **Distance barely matters:** near-zero GBT importance and a 0.02 correlation with the label.

## Limitations

- **Data loss in cleaning:** the cancelled-flight filter removed 31% of rows, far above typical cancellation rates, and the null drop removed about half of the sample. Both need to be checked.
- **Threshold picked on the test set:** the threshold-tuned metrics are somewhat optimistic.
- **GBT trained without class weights:** only LR and RF used `weightCol`.
- **No weather or air traffic control data** in the feature set.

Full details: [docs/analysis.md](docs/analysis.md)

## Repository structure

```
├── README.md
├── requirements.txt
├── data/README.md          # dataset source and columns used (raw data not included)
├── docs/analysis.md        # detailed tables, limitations, next steps, EDA figures
├── figures/                # plots saved from the notebook
└── notebooks/flight_delay_prediction_pyspark.ipynb
```

## How to run

1. Download `flight_data_2024.csv` from [Kaggle](https://www.kaggle.com/datasets/hrishitpatil/flight-data-2024) (see [data/README.md](data/README.md)).
2. Upload it to Google Drive at `MyDrive/DSAI4202_Project/flight_data_2024.csv`, or change the path in the data loading cell.
3. Open the notebook in Google Colab and run the cells from top to bottom.

The saved outputs come from a run on Spark 4.0.2. Cells were not run strictly in order, so a fresh run may differ slightly.

## Requirements

`pip install -r requirements.txt` (pyspark, pandas, numpy, matplotlib, seaborn). Outside Colab you also need Java 17 or later, and you need to replace the Google Drive path.
