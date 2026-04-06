<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=2,12,24&height=200&section=header&text=🚢%20Titanic%20Survival%20Prediction&fontSize=42&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=Binary%20Classification%20with%20Decision%20Tree%20Cost%20Complexity%20Pruning&descAlignY=60&descAlign=50" width="100%"/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Seaborn](https://img.shields.io/badge/Seaborn-Dataset-4C72B0?style=for-the-badge&logo=python&logoColor=white)](https://seaborn.pydata.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

---

## 📌 Project Overview

The Titanic disaster is one of the most well-known datasets in data science. This project builds a **binary classification model** to predict passenger survival using Decision Tree classifiers. The key focus is on demonstrating how **Cost Complexity Pruning (CCP)** significantly reduces overfitting and improves test accuracy.

- **Task:** Binary Classification (Survived / Did Not Survive)
- **Dataset:** Titanic (via Seaborn built-in dataset)
- **Goal:** Maximize prediction accuracy while avoiding an overfitted tree

---

## 📂 Dataset

| Property | Details |
|:---|:---|
| Source | Seaborn built-in (`sns.load_dataset("titanic")`) |
| Total Samples | 891 passengers |
| Selected Features | 5 (`pclass`, `sex`, `age`, `fare`, `embarked`) |
| Target | `survived` (0 = Died, 1 = Survived) |
| Missing Values | `age` (177), `embarked` (2) — handled via imputation |

---

## 🔄 Pipeline Workflow

```
Raw Data → Missing Value Imputation → Label Encoding → Train/Test Split → Model Training → Pruning (CCP) → Evaluation
```

1️⃣ **Data Loading** — Titanic dataset loaded directly from Seaborn

2️⃣ **Preprocessing** — `age` filled with median; `embarked` filled with most frequent value

3️⃣ **Encoding** — `sex` and `embarked` label-encoded to numeric values

4️⃣ **Feature Selection** — 5 features: `pclass`, `sex`, `age`, `fare`, `embarked`

5️⃣ **Baseline Model** — Full Decision Tree (no pruning) trained and evaluated

6️⃣ **Cost Complexity Pruning** — `cost_complexity_pruning_path` used to extract all alpha values; best alpha selected via grid search

7️⃣ **Best Model** — Retrained with optimal `ccp_alpha = 0.00154` for a cleaner, generalizable tree

---

## 🤖 Models

### 1️⃣ Decision Tree — No Pruning (Baseline)

```python
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
```

- Fully grown tree, no depth limit
- High variance — prone to overfitting training data
- Accuracy: **76.54%** on test set

---

### 2️⃣ Decision Tree — Cost Complexity Pruning ⭐ Best Model

```python
path = full_tree.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas

# Grid search for best alpha
best_model = DecisionTreeClassifier(ccp_alpha=0.0015407)
best_model.fit(X_train, y_train)
```

- Iterates over all pruning alphas to find optimal complexity
- Reduces tree depth — fewer splits, less noise sensitivity
- Accuracy: **83.80%** on test set

---

## 📊 Results

| Model | Accuracy | Notes |
|:---|:---:|:---|
| Decision Tree (No Pruning) | 76.54% | Overfit baseline |
| 🏆 **Decision Tree (CCP Pruned)** | **83.80%** | Best model — α = 0.00154 |

> Pruning delivered a **+7.3% accuracy improvement** over the unpruned baseline.

---

## 🔍 Key Insights

- 🌲 **Unpruned Decision Trees overfit** — adding all splits memorizes noise rather than learning patterns
- ✂️ **CCP Pruning** with `ccp_alpha = 0.00154` was the sweet spot — pruned enough branches to generalize without losing discriminative splits
- 🚺 **`sex`** was the most informative feature — consistent with historical "women and children first" evacuation behavior
- 💰 **`fare`** and **`pclass`** act as proxies for socioeconomic status, which strongly correlates with lifeboat access
- 📉 The best pruned model improved accuracy by **7.3 percentage points** over the baseline

---

## 🗂️ Repository Structure

```
titanic-survival-prediction/
│
├── titanic_survived.ipynb     # Main notebook with full pipeline
└── README.md                  # Project documentation
```

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/ronakrajput8882/Titanic-Survived-Died-Predictor.git
cd Titanic-Survived-Died-Predictor

# Install dependencies
pip install pandas scikit-learn seaborn matplotlib jupyter

# Launch the notebook
jupyter notebook titanic_survived.ipynb
```

---

## 🧠 Key Learnings

- Decision Trees are intuitive but suffer from high variance when fully grown
- Cost Complexity Pruning (CCP) provides a principled, data-driven way to reduce overfitting
- Imputing with median (continuous) vs most frequent (categorical) is an important preprocessing choice
- Label Encoding is sufficient for ordinal and binary categorical features in tree-based models
- Iterating over all alpha values to find the best pruning threshold is an effective model selection strategy

---

## 🛠️ Tech Stack

| Tool | Use |
|:---|:---|
| Python 3.10+ | Core language |
| Pandas | Data manipulation |
| Seaborn | Dataset loading & visualization |
| Matplotlib | Tree visualization |
| Scikit-learn | Modeling, imputation, encoding, evaluation |
| Jupyter Notebook | Interactive development |

---

<div align="center">

### Connect with me

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/ronakrajput8882)
[![Instagram](https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white)](https://instagram.com/techwithronak)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ronakrajput8882)

*If you found this useful, please ⭐ the repo!*

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=2,12,24&height=100&section=footer" width="100%"/>

</div>
