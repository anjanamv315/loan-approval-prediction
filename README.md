# Comparative Analysis of Machine Learning Algorithms for Loan Approval Prediction Using Python

**An Academic Micro Project for MTech in Artificial Intelligence & Data Science**

---

## 1. Abstract
In modern retail banking, evaluating loan applications quickly and accurately while minimizing default risk is a fundamental challenge. Traditional credit assessment methods frequently rely on rigid scorecards and subjective human underwriting, leading to operational delays, human error, and inconsistent approvals. This micro project presents an end-to-end, reproducible machine learning comparative framework developed using Python, scikit-learn, and Flask. Three supervised classification algorithms—**Logistic Regression**, **Decision Tree Classifier**, and **Random Forest Classifier**—are evaluated on the benchmark Loan Prediction dataset. Preprocessing, missing-value imputation, categorical encoding, and feature scaling are encapsulated strictly within scikit-learn `Pipeline` objects to prevent data leakage. Using 5-fold Stratified Cross-Validation on the training partition (75%), **Random Forest Classifier** is selected as the top-performing model with a mean cross-validation F1-score of 0.8638. On the held-out test partition (25%, N=154), the models achieve robust generalization (Logistic Regression: 86.36% accuracy, Random Forest: 85.71% accuracy, Decision Tree: 79.22% accuracy). The project includes a modern, responsive web application backed by SQLite to provide real-time multi-model inference, probability scoring, diagnostic visualization, and persistent application auditing.

---

## 2. Introduction & Problem Statement
Lending institutions process thousands of loan applications annually. Assessing an applicant's creditworthiness requires evaluating multi-dimensional applicant attributes, including demographic background, income levels, loan amount, repayment term, and past credit history. 

### Problem Statement
How can a machine learning system reliably predict loan approval status while providing an empirical, unbiased performance comparison across fundamentally distinct classification paradigms (linear models, rule-based trees, and bagged ensembles) without data leakage or overfitting?

---

## 3. Motivation & Objectives
### Motivation
Machine learning can uncover non-linear relationships and interactions within historical borrower data that standard rule thresholds miss. However, for academic rigor, models must not be selected using biased test scores, and predictions must remain explainable and transparent.

### Primary Objectives
1. Implement a clean, leak-free scikit-learn preprocessing and modeling pipeline.
2. Train and evaluate three supervised machine learning algorithms:
   - Logistic Regression
   - Decision Tree Classifier
   - Random Forest Classifier
3. Employ 5-fold Stratified Cross-Validation strictly on training data for model selection.
4. Evaluate all three trained models on a held-out test set using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
5. Identify the best model using a predetermined selection criterion (Highest CV F1-score).
6. Build a complete, responsive Flask web application with SQLite logging and dynamic Matplotlib visualization.

---

## 4. Scope & System Comparison
### Existing System vs. Proposed System
| Aspect | Traditional / Existing System | Proposed Machine Learning System |
|---|---|---|
| **Underwriting Methodology** | Manual credit officer review and rigid cutoff scorecards. | Statistical classification pipelines evaluating multidimensional patterns. |
| **Model Evaluation** | Single algorithm assessed on random train/test splits without CV. | 5-Fold Stratified Cross-Validation on training data + Held-out testing. |
| **Data Leakage Risk** | High; imputation and scaling often applied on the full dataset. | Zero; all transformations encapsulated within scikit-learn `Pipeline`. |
| **Multi-Model Inference** | Single model output without cross-verification. | Parallel predictions from all three models displayed side-by-side. |
| **Persistence & Audit** | Ephemeral or external proprietary spreadsheets. | Embedded SQLite database archiving applicant parameters and predictions. |

---

## 5. System Architecture
```
                  +----------------------------------+
                  |           End User               |
                  +-----------------+----------------+
                                    |
                                    v [HTTP Request]
                  +----------------------------------+
                  |     Flask Web Server (app.py)    |
                  +-----------------+----------------+
                                    |
      +-----------------------------+-----------------------------+
      |                             |                             |
      v                             v                             v
[Home Route: /]           [Predict Route: /predict]     [Comparison: /comparison]
- Project Scope           - Server-side Validation      - Loads models/metrics.json
- Workflow Summary        - DataFrame Construction      - Displays CV & Test tables
- Selected Model Badge    - Multi-Model Inference       - Renders static/comparison.png
                                    |
      +-----------------------------+-----------------------------+
      |                             |                             |
      v                             v                             v
[models/loan_models.joblib]   [SQLite DB: loan_history.db]  [templates/result.html]
- Logistic Regression Pipe    - Parameterized INSERT        - Multi-Model Verdicts
- Decision Tree Pipe          - Stores inputs & outputs     - Estimated Probabilities
- Random Forest Pipe          - Reverse-chronological log   - Demographic Summary
```

---

## 6. Technology Stack
- **Language:** Python 3.12 / 3.x
- **Web Framework:** Flask (Jinja2 templating engine)
- **Machine Learning:** scikit-learn (`Pipeline`, `ColumnTransformer`, `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`)
- **Data Manipulation:** pandas, NumPy
- **Visualization:** Matplotlib (`Agg` non-interactive backend)
- **Serialization:** joblib, JSON
- **Database:** SQLite3
- **Frontend UI:** HTML5, CSS3, Bootstrap 5.3 (via CDN), Bootstrap Icons

---

## 7. Dataset Description
- **File Name:** `loan_data_set.csv`
- **Location:** `dataset/loan_data_set.csv`
- **Total Records:** 614 rows &times; 13 columns
- **Target Variable:** `Loan_Status` (`Y` = Approved [422 records, 68.7%], `N` = Rejected [192 records, 31.3%])

### Feature Attributes
| Column Name | Type | Description | Handling Strategy |
|---|---|---|---|
| `Loan_ID` | String | Unique application identifier | Excluded from training features |
| `Gender` | Categorical | Applicant gender (`Male`, `Female`) | Mode imputation, One-Hot Encoding |
| `Married` | Categorical | Marital status (`Yes`, `No`) | Mode imputation, One-Hot Encoding |
| `Dependents` | Categorical | Number of dependents (`0`, `1`, `2`, `3+`) | Cleaned `'3+'` &rarr; `'3'`, Mode imputation, One-Hot Encoding |
| `Education` | Categorical | Education level (`Graduate`, `Not Graduate`) | One-Hot Encoding |
| `Self_Employed` | Categorical | Employment status (`Yes`, `No`) | Mode imputation, One-Hot Encoding |
| `ApplicantIncome` | Numeric | Monthly applicant income | Median imputation, StandardScaler |
| `CoapplicantIncome`| Numeric | Monthly co-applicant income | Median imputation, StandardScaler |
| `LoanAmount` | Numeric | Loan amount in thousands | Median imputation, StandardScaler |
| `Loan_Amount_Term` | Numeric | Loan term in months (e.g. 360) | Median imputation, StandardScaler |
| `Credit_History` | Numeric | Credit guidelines met (`1.0` = Yes, `0.0` = No)| Median imputation, StandardScaler |
| `Property_Area` | Categorical | Property location (`Urban`, `Semiurban`, `Rural`) | One-Hot Encoding |
| `Loan_Status` | Target | Approval decision (`Y`, `N`) | Mapped to binary integer (`1`, `0`) |

---

## 8. Data Preprocessing & Leakage Prevention
1. **Target Separation:** `Loan_Status` is isolated and mapped to binary targets ($Y \rightarrow 1$, $N \rightarrow 0$). `Loan_ID` is removed.
2. **Train/Test Stratification:** Data is split into 75% training (460 rows) and 25% held-out test (154 rows) using `stratify=y` to preserve the 68.7% / 31.3% class ratio.
3. **Leak-Free Transformation:**
   - Numerical Transformer: `SimpleImputer(strategy='median')` followed by `StandardScaler()`.
   - Categorical Transformer: `SimpleImputer(strategy='most_frequent')` followed by `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
   - All transformers are fitted **strictly on training partitions** inside the scikit-learn `Pipeline`. Test data and incoming web inference inputs are only transformed, never fitted.

---

## 9. Machine Learning Algorithms
### A. Logistic Regression
- **Mathematical Principle:** Models log-odds using the Sigmoid link function: $P(Y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta^T X)}}$.
- **Advantages:** Computationally efficient, highly interpretable coefficients, strong baseline for linearly separable features.
- **Limitations:** Cannot capture complex feature interactions without manual feature engineering; sensitive to extreme outliers.

### B. Decision Tree Classifier
- **Mathematical Principle:** Recursively splits data partitions at feature thresholds to maximize Gini impurity reduction ($Gini = 1 - \sum p_i^2$).
- **Advantages:** Captures non-linear relationships and thresholds naturally; easy to inspect as logical if-else rules.
- **Limitations:** High variance; prone to overfitting on small training samples. Controlled in this project via `max_depth=4` and `min_samples_leaf=5`.

### C. Random Forest Classifier
- **Mathematical Principle:** Ensemble of 100 decorrelated decision trees built on bootstrap samples (bagging) with randomized feature subsets at each split. Final prediction is calculated via majority vote.
- **Advantages:** Substantially reduces model variance while maintaining low bias; robust against noise and outliers.
- **Limitations:** Higher computational overhead than single trees; less directly interpretable than a single linear equation.

---

## 10. Experimental Results & Model Selection
### A. 5-Fold Stratified Cross-Validation on Training Set (N=460)
*Model selection was performed exclusively using cross-validation on training data.*

| Algorithm Name | Mean CV Accuracy | Mean CV Precision (Y) | Mean CV Recall (Y) | Mean CV F1-Score (Y) | Status |
|---|---|---|---|---|---|
| **Logistic Regression** | 78.70% (&plusmn;0.028) | 77.46% (&plusmn;0.026) | 97.47% (&plusmn;0.013) | 0.8629 (&plusmn;0.017) | Candidate |
| **Decision Tree** | 74.57% (&plusmn;0.038) | 76.83% (&plusmn;0.022) | 90.17% (&plusmn;0.052) | 0.8291 (&plusmn;0.029) | Candidate |
| **Random Forest** | **78.70%** (&plusmn;0.029) | **77.25%** (&plusmn;0.030) | **98.11%** (&plusmn;0.012) | **0.8638** (&plusmn;0.015) | **Selected Best Model** |

**Selection Rationale:** Random Forest achieved the highest cross-validation F1-score (0.8638) and superior cross-validation recall (98.11%), demonstrating the strongest generalization among all candidates on the training folds.

### B. Held-Out Test Set Final Evaluation (N=154)
*Evaluated only once after model selection was completed.*

| Algorithm Name | Accuracy | Precision (Approved) | Recall (Approved) | F1-Score (Approved) | Precision (Rejected) | Recall (Rejected) | F1-Score (Rejected) | Macro F1 |
|---|---|---|---|---|---|---|---|---|
| **Logistic Regression** | 86.36% | 84.00% | 99.06% | 0.9091 | 96.55% | 58.33% | 0.7273 | 0.8182 |
| **Decision Tree** | 79.22% | 82.46% | 88.68% | 0.8545 | 70.00% | 58.33% | 0.6364 | 0.7455 |
| **Random Forest** | 85.71% | 83.33% | 99.06% | 0.9052 | 96.43% | 56.25% | 0.7105 | 0.8078 |

### C. Confusion Matrices on Held-out Test Set (N=154: 48 Rejected, 106 Approved)
- **Logistic Regression:** TN=28, FP=20, FN=1, TP=105
- **Decision Tree:** TN=28, FP=20, FN=12, TP=94
- **Random Forest:** TN=27, FP=21, FN=1, TP=105

---

## 11. Folder Structure
```
LoanApprovalProject/
│
├── app.py                     # Flask web server, route definitions, DB integration
├── train_model.py             # ML pipeline, cross-validation, evaluation & chart generator
├── requirements.txt           # Python dependencies
├── README.md                  # Comprehensive project documentation
├── .gitignore                 # Version control exclusions
│
├── dataset/
│   └── loan_data_set.csv      # Standard benchmark loan dataset (614 rows)
│
├── models/
│   ├── loan_models.joblib     # Serialized pipeline bundle & metadata
│   └── metrics.json           # Statistical cross-validation & test results
│
├── static/
│   ├── style.css              # Custom styling & academic theme
│   └── comparison.png         # Comparative Matplotlib visualization chart
│
├── templates/
│   ├── base.html              # Shared layout, navbar, footer & flash alerts
│   ├── index.html             # Project landing page & architecture overview
│   ├── predict.html           # Interactive loan prediction form
│   ├── result.html            # Multi-algorithm prediction outputs & probabilities
│   ├── comparison.html        # Detailed metric tables, confusion matrices & chart
│   └── history.html           # SQLite prediction history audit table
│
└── loan_history.db            # Embedded SQLite database (generated automatically)
```

---

## 12. Installation & Execution Guide (Windows & VS Code)

### Step 1: Verify Python Installation
Open PowerShell in Visual Studio Code and verify Python:
```powershell
python --version
```
*Expected output:* Python 3.10.x, 3.11.x, or 3.12.x.

### Step 2: Create and Activate Virtual Environment
```powershell
# Create virtual environment
python -m venv venv

# Activate in PowerShell
.\venv\Scripts\Activate.ps1

# (Alternative for Command Prompt)
# .\venv\Scripts\activate.bat
```

### Step 3: Install Required Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Verify Dataset
Ensure that `dataset/loan_data_set.csv` is present. If missing, it can be downloaded directly from:
`https://raw.githubusercontent.com/atulpatelDS/Data_Files/master/Loan_Dataset/loan_data_set.csv`

### Step 5: Execute Model Training & Evaluation
```powershell
python train_model.py
```
*Expected terminal output:*
```text
================================================================================
LOAN APPROVAL PREDICTION - MACHINE LEARNING PIPELINE & MODEL COMPARISON
================================================================================
[1/7] Checking dataset availability at: ...\dataset\loan_data_set.csv
      Successfully loaded CSV. Raw dataset shape: 614 rows, 13 columns.
...
[4/7] Performing 5-Fold Stratified Cross-Validation on Training Data...
      >>> SELECTED BEST MODEL: [Random Forest] <<<
[5/7] Fitting pipelines on full training data & evaluating on held-out test set...
[6/7] Generating comparative performance visualization at: ...\static\comparison.png
[7/7] Persisting model bundle and evaluation metrics...
================================================================================
 TRAINING & EVALUATION PIPELINE COMPLETED SUCCESSFULLY!
================================================================================
```

### Step 6: Start Flask Web Server
```powershell
python app.py
```
*Terminal output:*
```text
======================================================================
 LOAN APPROVAL PREDICTION FLASK WEB SERVER
 Local URL: http://127.0.0.1:5000
 Academic project: Strictly for educational analysis.
======================================================================
 * Running on http://127.0.0.1:5000
```

### Step 7: Open Application in Web Browser
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 13. Application Usage Walkthrough
1. **Home Page (`/`):** Review the project objectives, architecture, algorithm overviews, and current selected model.
2. **Predict Page (`/predict`):** Fill out applicant details:
   - *Example Profile A (Approved):* Male, Married=Yes, Dependents=1, Education=Graduate, Self_Employed=No, ApplicantIncome=6000, CoapplicantIncome=2000, LoanAmount=140 ($140k), Term=360, Credit_History=1.0 (Meets Guidelines), Property_Area=Semiurban.
   - *Example Profile B (Rejected):* Female, Married=No, Dependents=0, Education=Not Graduate, Self_Employed=Yes, ApplicantIncome=1500, CoapplicantIncome=0, LoanAmount=250 ($250k), Term=360, Credit_History=0.0 (Does Not Meet Guidelines), Property_Area=Rural.
3. **Results Page (`/predict` POST):** Review real-time predictions from Logistic Regression, Decision Tree, and Random Forest along with model-estimated probabilities and database record ID.
4. **Model Comparison (`/comparison`):** Examine the 5-fold cross-validation table, held-out test evaluation table, confusion matrices, and the Matplotlib visualization chart.
5. **Prediction History (`/history`):** Inspect previous submissions archived in reverse chronological order in `loan_history.db`.

---

## 14. Database Inspection Guide
The SQLite database `loan_history.db` can be inspected without external services using Python or VS Code extensions:
```powershell
python -c "import sqlite3; conn=sqlite3.connect('loan_history.db'); print(conn.execute('SELECT id, timestamp, applicant_income, selected_model_prediction FROM predictions').fetchall())"
```
Or install the **SQLite Viewer** extension in Visual Studio Code and click `loan_history.db`.

---

## 15. Ethical Considerations & Limitations
1. **Academic Purpose:** This project is designed exclusively for educational and research demonstration. It does not constitute a real lending engine.
2. **Historical Bias:** Historical credit datasets often reflect systemic disparities. If certain demographic groups faced historical rejection, standard models trained on historical data may propagate those patterns.
3. **Class Imbalance:** Approved loans outnumber rejected loans (68.7% vs 31.3%). While handled through stratified sampling and F1-score evaluation, real-world deployment requires cost-sensitive learning or threshold tuning.
4. **Statistical Likelihood vs. Credit Decisions:** Model probability outputs represent mathematical distance from decision boundaries, not a guaranteed credit outcome.

---

## 16. Future Enhancements
- Integration of gradient boosting algorithms (XGBoost, LightGBM, CatBoost).
- Implementation of Explainable AI (XAI) using SHAP (SHapley Additive exPlanations) and LIME to interpret individual feature contributions.
- Automated hyperparameter optimization using `Optuna` or `GridSearchCV`.
- Docker containerization for portable cloud deployment.

---

## 17. References
1. Dua, D. and Graff, C. (2019). UCI Machine Learning Repository. Irvine, CA: University of California, School of Information and Computer Science.
2. Analytics Vidhya: Loan Prediction Practice Problem Dataset.
3. Pedregosa et al., *Scikit-learn: Machine Learning in Python*, JMLR 12, pp. 2825-2830, 2011.
4. Grus, J., *Data Science from Scratch: First Principles with Python*, O'Reilly Media, 2019.
