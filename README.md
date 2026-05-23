# Credit Card Fraud Detection: Anomaly Detection with KMeans and IsolationForest

Unsupervised anomaly detection on 10,000 credit card transactions using two independent models.
No labels. No supervision. Just the data and a business threshold.

---

## The Problem

A financial stakeholder reports that roughly **0.4% of transactions are fraudulent** and 99.6% are valid.
There are no confirmed fraud labels in this dataset.

The task: identify the most anomalous transactions using unsupervised methods, calibrated to the stakeholder's estimate.

---

## Dataset

- **10,000** anonymized credit card transactions
- **29 features**: V1-V28 (PCA-transformed by the original data provider for anonymization) + `Amount`
- No target column. Purely unsupervised.
- Source: subset of the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## Approach

Both models run on all **29 scaled features**. No additional dimensionality reduction applied.

### Part 1: KMeans Clustering

- Scale all features with `StandardScaler`
- Fit KMeans with 3 clusters (`random_state=42`)
- Use `scipy.spatial.distance.cdist` to compute Euclidean distance from every transaction to every cluster center
- Take the minimum distance per transaction (distance to closest cluster)
- Flag the top 0.4% most distant points as anomalies (99.6th percentile threshold)

### Part 2: IsolationForest

- Fit directly on the 29 scaled features
- Set `contamination=0.004` to match the stakeholder estimate
- Flag transactions with a negative decision score

### Part 3: Model Agreement

- Compare which transactions both models flagged
- Transactions flagged by both are the highest-confidence fraud candidates

---

## Results

| | KMeans | IsolationForest |
|---|---|---|
| **Method** | Distance from cluster centroid | Isolation path length in random trees |
| **Input** | 29 scaled features | 29 scaled features |
| **Threshold** | 99.6th percentile | `contamination = 0.004` |
| **Flagged** | 40 (0.40%) | 40 (0.40%) |
| **Flagged by BOTH** | **27 (highest confidence)** | **27 (highest confidence)** |

Both models independently arrived at exactly 40 suspicious transactions, matching the business estimate precisely.
**27 of those 40 were flagged by both models**, making them the strongest fraud candidates in the dataset.

---

## KMeans: How the Detection Works

### Distance Distribution

After fitting 3 clusters, every transaction gets a distance score to its nearest centroid.
Normal transactions cluster tightly at low distances. Fraudulent ones sit far out.

![KMeans Distance Distribution](assets/kmeans_distance_distribution.png)

The bulk of transactions fall between distances **2 and 8**.
A tiny handful scatter out to **20, 40, even 65**. Those are the candidates.
The model does not define what fraud looks like. It finds what does not look normal.

---

### Anomaly Threshold

The 99.6th percentile of distances becomes the decision boundary.
Anything above it gets flagged.

![KMeans Anomaly Threshold](assets/kmeans_anomaly_threshold.png)

The threshold lands at **20.91**.
White area: 9,960 normal transactions.
Pink zone: 40 fraud candidates.
Some sit just above the line (borderline). Others stretch past distance 60 (no ambiguity there).

---

### Do the Flagged Transactions Look Different?

A model flagging random transactions is useless. This checks whether the flags make sense.

![KMeans Amount Distribution](assets/kmeans_amount_distribution.png)

The blue peak (normal) is a near-vertical spike at $0. Everyday spending.
The orange area (flagged) is flat and spread from **negative values to past $10,000**.

Two signals worth noting:
- Transactions above $2,000 appear in orange but almost never in blue. Large amounts are a classic fraud signal.
- Negative amounts appear in orange. These are refunds or chargebacks. Unusual refund patterns indicate a separate fraud type worth investigating independently.

---

## IsolationForest: How the Detection Works

### Anomaly Score Distribution

IsolationForest builds hundreds of random trees and measures how quickly each transaction
can be isolated from the rest. Unusual transactions isolate fast and get low (negative) scores.

![IsolationForest Score Distribution](assets/iforest_score_distribution.png)

Nearly all 10,000 transactions pile up between **+0.12 and +0.29** on the right.
Left of the red line (score = 0): roughly 40 transactions.
Notice the gap between 0 and -0.07, then a cluster near -0.08 to -0.12.
These are not close calls. The model is certain about them.

---

### Do the IsolationForest Flags Make Sense?

![IsolationForest Amount Distribution](assets/iforest_amount_distribution.png)

Same pattern as KMeans. Orange is flat, wide, spread from deeply negative to $10,000+.
Blue is a spike at zero.

Both distributions are nearly identical, which matters.
Two completely different algorithms, using completely different mathematics, flagging the same type of transaction.
That convergence is not noise.

---

## Model Comparison

```python
set_kmeans = set(idx_anomalies_kmeans)
set_iso    = set(idx_anomalies_iso)

overlap    = set_kmeans & set_iso   # 27 transactions
only_kmeans = set_kmeans - set_iso  # 13 transactions
only_iso    = set_iso - set_kmeans  # 13 transactions
```

27 transactions were flagged by both models independently.
Two detectives. Different methods. Same 27 transactions.

---

## Recommended Actions

1. **Immediate review: 27 transactions** flagged by both models.
   Two independent methods agree. These are the highest-confidence fraud candidates.

2. **Secondary review: 13 transactions** flagged by only one model.
   May be real fraud one model caught that the other missed, or legitimate edge cases.
   Human judgment required.

3. **Investigate the negative-amount cluster separately.**
   These likely represent a different fraud scheme (refund manipulation, chargeback fraud)
   and warrant a different investigative approach than the large-amount cases.

---

## Next Steps

If ground truth labels become available, the immediate next steps are:

- **Precision**: of the 40 flagged transactions, how many were confirmed fraud?
- **Recall**: of all confirmed fraud cases, how many did we catch?
- Use those metrics to tune the KMeans threshold and IsolationForest contamination parameter
- Consider adding a third model (e.g., Local Outlier Factor) to further strengthen the agreement signal

---

## Tech Stack

| Tool | Purpose |
|---|---|
| `pandas`, `numpy` | Data handling |
| `sklearn.preprocessing.StandardScaler` | Feature scaling |
| `sklearn.cluster.KMeans` | Clustering model |
| `scipy.spatial.distance.cdist` | Distance matrix computation |
| `sklearn.ensemble.IsolationForest` | Isolation-based anomaly detection |
| `matplotlib`, `seaborn` | Visualization |

---

## How to Run

```bash
git clone https://github.com/your-username/credit-card-fraud-detection.git
cd credit-card-fraud-detection
pip install -r requirements.txt
jupyter notebook credit_card_anomaly_detection.ipynb
```

The dataset loads automatically from a public Google Drive link in Cell 2. No local file needed.

---

## Project Structure

```
credit-card-fraud-detection/
├── credit_card_anomaly_detection.ipynb   # Main notebook
├── assets/
│   ├── kmeans_distance_distribution.png
│   ├── kmeans_anomaly_threshold.png
│   ├── kmeans_amount_distribution.png
│   ├── iforest_score_distribution.png
│   └── iforest_amount_distribution.png
└── README.md
```

---

## Author

Ali Abu Sohiban
Biotechnology graduate, Islamic University of Gaza.
Currently studying data science and machine learning.

[GitHub](https://github.com/your-username) | [LinkedIn](https://linkedin.com/in/your-profile)
