# Dataset Access Instructions

## Dataset

**Name:** Steam Reviews Dataset
**Author:** forgemaster
**Source:** https://www.kaggle.com/datasets/forgemaster/steam-reviews-dataset
**License:** Public dataset; no API key required. All data sourced from the publicly accessible Steam API.

---

## Description

This dataset contains user reviews scraped from the Steam Store via the Steam API. It includes review text, recommendation labels, playtime statistics, and user metadata for 8,183 games across 15,437,471 reviews from 6,976,390 unique users.

---

## Files

The dataset consists of 11 CSV files split by game appid range:

```
reviews-1-115.csv
reviews-115-1230.csv
reviews-1230-2345.csv
reviews-2345-4575.csv
reviews-4575-6805.csv
reviews-6805-9035.csv
reviews-9035-11265.csv
reviews-11265-13495.csv
reviews-13495-13500.csv
reviews-13500-13537.csv
reviews-13537-27075.csv
```

Total size: approximately 5.32 GB

---

## Columns

| Column | Type | Description |
|---|---|---|
| `steamid` | int | User ID |
| `appid` | int | Game ID |
| `voted_up` | bool | **Target variable.** True = positive recommendation, False = negative |
| `votes_up` | int | Number of users who found this review helpful |
| `votes_funny` | int | Number of users who found this review funny |
| `weighted_vote_score` | float | Helpfulness score computed by Steam |
| `playtime_forever` | int | Total hours played by user at time of retrieval |
| `playtime_at_review` | int | Hours played when the review was written |
| `num_games_owned` | int | Number of games owned by the reviewer |
| `num_reviews` | int | Number of reviews written by the reviewer |
| `review` | str | **Primary feature.** Full text of the written review |
| `unix_timestamp_created` | int | Review creation date (Unix timestamp) |
| `unix_timestamp_updated` | int | Review last updated date (Unix timestamp) |

---

## Download Instructions

1. Create a free Kaggle account at https://www.kaggle.com if you do not have one.
2. Navigate to https://www.kaggle.com/datasets/forgemaster/steam-reviews-dataset
3. Click **Download** to download the full dataset zip, or use the Kaggle CLI:
   ```
   kaggle datasets download -d forgemaster/steam-reviews-dataset
   ```
4. Extract the zip and place all 11 CSV files into:
   ```
   data/raw/
   ```

---

## How This Project Uses the Data

The full 5.32 GB dataset is not used directly. The notebook samples approximately 100,000 reviews proportionally across all 11 files, stratified by `voted_up`, and saves the working dataset to `data/processed/`. A 1,000-row sample is saved to `data/sample/` for quick verification without downloading the full dataset.

Only three columns are retained: `review` (primary feature), `voted_up` (target variable), and `appid` (retained for error analysis only, not used as a model feature). All other metadata columns are dropped at the sampling step.
