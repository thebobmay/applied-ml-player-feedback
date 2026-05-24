# Machine Learning Classification of Player Feedback from Steam Reviews

Udacity AI Masters Capstone - Project 3: Applied Machine Learning

**Student:** Robert Mayfield
**Series:** AI Game Director Studio (Project 3 of 7)

---

## Project Description

This project builds a supervised machine learning classifier that predicts whether a Steam game review is positive or negative. The primary intended application is playtesting support: automatically classifying written player feedback to help development teams identify negative reception signals without manually reviewing every response.

The workflow covers dataset loading and inspection, text preprocessing, TF-IDF vectorization, model training and selection, evaluation with classification metrics, error analysis, and discussion of limitations and responsible use.

---

## Dataset

**Name:** Steam Reviews Dataset
**Source:** Kaggle - forgemaster/steam-reviews-dataset
**Format:** CSV
**Target variable:** `voted_up` (binary: positive or negative recommendation)
**Size:** ~5.32 GB across 11 CSV files (15,437,471 reviews)

See `dataset_access_instructions.md` for full download instructions.

---

## Files Included

```
notebooks/modeling.ipynb                         main project notebook
reports/Machine_Learning_Analysis_Report.pdf     full written analysis report
reports/module_summary.pdf                       identical copy of report
requirements.txt                         pinned Python dependencies
dataset_access_instructions.md           dataset source and download steps
src/preprocessing.py                     text cleaning and TF-IDF vectorization functions
src/evaluation.py                        model evaluation function
src/inspection.py                        dataframe inspection utility
data/sample/                             small sample for quick review
outputs/figures/                         saved chart files
outputs/tables/model_metrics.csv         evaluation metrics summary
outputs/model_artifacts/                 saved model and vectorizer for Project 7
```

---

## How to Run

1. Create and activate a Python environment (Python 3.13.3 was used for this project):
  ```
   python -m venv .venv
  ```
   On Windows:
   On macOS or Linux:
2. Install dependencies:
  ```
   pip install -r requirements.txt
  ```
   Note: `aif360` (IBM AI Fairness 360) is used for fairness evaluation and installs via pip without additional configuration for the metrics used in this project.
3. Download the Steam Reviews Dataset from Kaggle (see `dataset_access_instructions.md`) and place the `archive.zip` file (do not extract it) into `data/raw/`. The notebook reads the archive directly.  See dataset_access_instructions.md for more details.
4. Open the notebook in Jupyter:
  ```
   jupyter notebook notebooks/modeling.ipynb
  ```
5. Run all cells from top to bottom.

---

## Bias and Responsible Data Handling

Steam reviews skew toward English language PC gaming demographics and toward games with large user bases. The dataset is class imbalanced (approximately 85% positive). Class weighting is applied during training to mitigate this. The classifier is designed as an advisory signal and should not be used for automated decision-making without human oversight.

---

## Future Integration Reflection

### How this classifier could support the AI Game Director Studio

The trained classifier converts raw review text into positive/negative reception signals that the AI Game Director can use to surface games receiving unexpected negative feedback, segment player sentiment over time, and flag issues emerging from playtester written responses without requiring manual review at scale.

### How this dataset and model would need to evolve for deeper integration

The model is trained on post launch consumer reviews from a general Steam audience. For playtester applications it would need fine tuning on labeled playtester feedback collected during closed or internal testing phases, which does not currently exist in a public form. Playtester language, focus areas, and feedback style differ meaningfully from public consumer reviews.

### How agentic automation could assist this workflow

An agentic pipeline could automate review scraping, batch classification, trend aggregation, and anomaly surfacing on a scheduled basis, feeding structured sentiment signals directly into the Game Director system.

---

## Requirements

See `requirements.txt` for the full pinned dependency list.

Key libraries: Python 3.13.3, pandas, numpy, scikit-learn, matplotlib, aif360, jupyter