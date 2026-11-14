# 📦 Late Delivery Risk Predictor (Olist)

Predicts, at purchase time, whether an e-commerce order will arrive after its estimated delivery date, so operations can warn customers and chase carriers early.

🔗 **Live app:** https://olist-late-delivery-xbkbh4tbpben3xq2nf4gm9.streamlit.app

![App screenshot](screenshot.png)

## Results

Test set: 19,294 orders (20% hold-out, stratified). Late orders are about 8% of the data.

| Model | CV PR-AUC | CV ROC-AUC |
|---|---|---|
| Baseline (majority class) | 0.081 | 0.500 |
| Logistic Regression | 0.183 | 0.708 |
| Random Forest | 0.275 | 0.777 |
| Hist Gradient Boosting (final, tuned) | 0.285 | n/a |

Final model on the untouched test set: **PR-AUC 0.293, ROC-AUC 0.787**.

How the business can use it (test set):

| Review the riskiest | Share of late orders caught | Late rate inside the group |
|---|---|---|
| Top 5% | 25% | 40% |
| Top 10% | 40% | 32% |
| Top 20% | 59% | 24% |

## Limitations
- One marketplace, 2016 to 2018.
- On a time-based split (train on older orders, test on newer), the model is weaker: PR-AUC 0.102, about 1.9x better than chance, versus about 3.6x on the random split. The purchase month is the top feature, but late spikes happened in specific periods (e.g. Nov 2017, Feb to Mar 2018) and may not repeat.
- The model does not see seller or carrier problems that happen after purchase.
- Distance in the app is approximated from state centers.

## Data
[Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (Kaggle). The raw data is not included here.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tools
Python, pandas, scikit-learn (Pipeline, HistGradientBoosting), Streamlit.
