"""
================================================================================
Comparative Analysis of Machine Learning Algorithms for Loan Approval Prediction
File: train_model.py
Purpose: Complete ML training pipeline, cross-validation model selection,
         held-out test evaluation, metric serialization, and chart generation.
Author: Academic MTech AI & Data Science Project
================================================================================
"""

import os
import sys
import json
import joblib
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server/CLI chart generation
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ------------------------------------------------------------------------------
# 1. PATH CONFIGURATION & DIRECTORY SETUP
# ------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'loan_data_set.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
STATIC_DIR = os.path.join(BASE_DIR, 'static')
MODEL_BUNDLE_PATH = os.path.join(MODELS_DIR, 'loan_models.joblib')
METRICS_JSON_PATH = os.path.join(MODELS_DIR, 'metrics.json')
COMPARISON_CHART_PATH = os.path.join(STATIC_DIR, 'comparison.png')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# ------------------------------------------------------------------------------
# 2. DATASET VALIDATION & SCHEMA SPECIFICATION
# ------------------------------------------------------------------------------
EXPECTED_COLUMNS = [
    'Loan_ID', 'Gender', 'Married', 'Dependents', 'Education',
    'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome',
    'LoanAmount', 'Loan_Amount_Term', 'Credit_History',
    'Property_Area', 'Loan_Status'
]

NUMERIC_FEATURES = [
    'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
    'Loan_Amount_Term', 'Credit_History'
]

CATEGORICAL_FEATURES = [
    'Gender', 'Married', 'Dependents', 'Education',
    'Self_Employed', 'Property_Area'
]

TARGET_COLUMN = 'Loan_Status'

def load_and_validate_dataset(csv_path: str) -> pd.DataFrame:
    """
    Validates CSV existence, column schema, non-emptiness, target integrity,
    and returns a clean pandas DataFrame.
    """
    print(f"[1/7] Checking dataset availability at: {csv_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"ERROR: Dataset not found at '{csv_path}'.\n"
            f"Please download 'loan_data_set.csv' and place it inside the 'dataset/' folder.\n"
            f"Reliable source: https://raw.githubusercontent.com/atulpatelDS/Data_Files/master/Loan_Dataset/loan_data_set.csv"
        )
    
    df = pd.read_csv(csv_path)
    print(f"      Successfully loaded CSV. Raw dataset shape: {df.shape[0]} rows, {df.shape[1]} columns.")
    
    # Check required columns
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"ERROR: Dataset is missing required columns: {missing_cols}")
    
    # Check for empty dataframe
    if df.empty or len(df) < 50:
        raise ValueError("ERROR: Dataset has fewer than 50 rows. Insufficient data for ML training.")
    
    # Validate target values
    unique_targets = set(df[TARGET_COLUMN].dropna().unique())
    if not unique_targets.issubset({'Y', 'N'}):
        raise ValueError(f"ERROR: Unexpected values in target column '{TARGET_COLUMN}': {unique_targets}. Expected {'Y', 'N'}.")
    
    # Inspect duplicate rows (excluding Loan_ID)
    duplicate_count = df.drop(columns=['Loan_ID']).duplicated().sum()
    print(f"      Duplicates detected (excluding Loan_ID): {duplicate_count}")
    
    # Report missing values
    print("      Missing value summary:")
    null_counts = df.isnull().sum()
    for col, count in null_counts.items():
        if count > 0:
            print(f"        - {col}: {count} missing values ({count/len(df)*100:.1f}%)")
    
    return df

# ------------------------------------------------------------------------------
# 3. FEATURE PREPROCESSING & STRATIFIED SPLIT
# ------------------------------------------------------------------------------
def prepare_data(df: pd.DataFrame):
    """
    Cleans Dependents '3+' to '3', extracts predictive features X and target y,
    and performs a stratified 75/25 train-test split.
    """
    print("[2/7] Preprocessing and partitioning data...")
    df_clean = df.copy()
    
    # Standardize 'Dependents' value '3+' to '3'
    df_clean['Dependents'] = df_clean['Dependents'].astype(str).str.strip().replace({'3+': '3', 'nan': np.nan})
    
    # Separate predictive features and target variable
    X = df_clean[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y_raw = df_clean[TARGET_COLUMN].copy()
    
    # Map target: Y -> 1 (Approved), N -> 0 (Rejected)
    y = y_raw.map({'Y': 1, 'N': 0}).astype(int)
    
    # Class distribution
    approved_count = (y == 1).sum()
    rejected_count = (y == 0).sum()
    print(f"      Class distribution: Approved (Y=1): {approved_count} ({approved_count/len(y)*100:.1f}%), "
          f"Rejected (N=0): {rejected_count} ({rejected_count/len(y)*100:.1f}%)")
    
    # Stratified 75% train / 25% test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )
    print(f"      Training set size: {X_train.shape[0]} samples | Held-out test set size: {X_test.shape[0]} samples")
    return X_train, X_test, y_train, y_test

# ------------------------------------------------------------------------------
# 4. PIPELINE CREATION (LEAK-FREE PREPROCESSING + ESTIMATOR)
# ------------------------------------------------------------------------------
def create_model_pipelines():
    """
    Constructs separate, leak-free scikit-learn Pipelines for:
    1. Logistic Regression (scaled numeric + one-hot encoded categorical)
    2. Decision Tree Classifier
    3. Random Forest Classifier
    All imputations and encodings are encapsulated strictly inside the pipeline.
    """
    print("[3/7] Building scikit-learn ColumnTransformer & Model Pipelines...")
    
    # Numerical pipeline: Median imputation + Standard Scaling
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical pipeline: Mode imputation + One-Hot Encoding (handle_unknown='ignore')
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine transformations
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, NUMERIC_FEATURES),
        ('cat', cat_transformer, CATEGORICAL_FEATURES)
    ])
    
    # Define three competitive algorithms with reproducible random_states
    pipelines = {
        'Logistic Regression': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(max_iter=1000, C=1.0, random_state=42))
        ]),
        'Decision Tree': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', DecisionTreeClassifier(max_depth=4, min_samples_split=10, min_samples_leaf=5, random_state=42))
        ]),
        'Random Forest': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=8, min_samples_leaf=4, random_state=42))
        ])
    }
    return pipelines

# ------------------------------------------------------------------------------
# 5. CROSS-VALIDATION & MODEL SELECTION (STRICTLY ON TRAINING SET)
# ------------------------------------------------------------------------------
def perform_model_selection(pipelines: dict, X_train: pd.DataFrame, y_train: pd.Series):
    """
    Performs 5-fold Stratified Cross-Validation strictly on X_train.
    Selects best model based on:
      Primary criterion: Highest mean Cross-Validation F1-score (positive class Y=1).
      Tie-breaker: Highest mean Cross-Validation Accuracy.
    Does NOT use the held-out test set for selection, preventing data leakage.
    """
    print("[4/7] Performing 5-Fold Stratified Cross-Validation on Training Data...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ['accuracy', 'precision', 'recall', 'f1']
    
    cv_results = {}
    best_model_name = None
    best_f1 = -1.0
    best_acc = -1.0
    
    for name, pipe in pipelines.items():
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        mean_acc = float(np.mean(scores['test_accuracy']))
        std_acc = float(np.std(scores['test_accuracy']))
        mean_prec = float(np.mean(scores['test_precision']))
        std_prec = float(np.std(scores['test_precision']))
        mean_rec = float(np.mean(scores['test_recall']))
        std_rec = float(np.std(scores['test_recall']))
        mean_f1 = float(np.mean(scores['test_f1']))
        std_f1 = float(np.std(scores['test_f1']))
        
        cv_results[name] = {
            'accuracy_mean': round(mean_acc, 4),
            'accuracy_std': round(std_acc, 4),
            'precision_mean': round(mean_prec, 4),
            'precision_std': round(std_prec, 4),
            'recall_mean': round(mean_rec, 4),
            'recall_std': round(std_rec, 4),
            'f1_mean': round(mean_f1, 4),
            'f1_std': round(std_f1, 4)
        }
        
        print(f"      -> {name}:")
        print(f"         CV Accuracy:  {mean_acc:.4f} (+/- {std_acc:.4f})")
        print(f"         CV Precision: {mean_prec:.4f} (+/- {std_prec:.4f})")
        print(f"         CV Recall:    {mean_rec:.4f} (+/- {std_rec:.4f})")
        print(f"         CV F1-score:  {mean_f1:.4f} (+/- {std_f1:.4f})")
        
        # Model selection logic
        if (mean_f1 > best_f1) or (abs(mean_f1 - best_f1) < 1e-4 and mean_acc > best_acc):
            best_f1 = mean_f1
            best_acc = mean_acc
            best_model_name = name
            
    print(f"\n      >>> SELECTED BEST MODEL: [{best_model_name}] <<<")
    print(f"      Selection Rationale: Achieved highest 5-Fold Cross-Validation F1-Score ({best_f1:.4f}) "
          f"and Cross-Validation Accuracy ({best_acc:.4f}) on the training partition.\n")
    
    return cv_results, best_model_name

# ------------------------------------------------------------------------------
# 6. FINAL TRAINING & HELD-OUT TEST EVALUATION
# ------------------------------------------------------------------------------
def evaluate_on_held_out_test(pipelines: dict, X_train: pd.DataFrame, y_train: pd.Series,
                              X_test: pd.DataFrame, y_test: pd.Series):
    """
    Fits all pipelines on full training set and evaluates once on the held-out test set.
    Calculates Accuracy, Precision, Recall, F1, Confusion Matrix, and Classification Report.
    """
    print("[5/7] Fitting pipelines on full training data & evaluating on held-out test set...")
    test_results = {}
    
    for name, pipe in pipelines.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        
        acc = float(accuracy_score(y_test, y_pred))
        prec_approved = float(precision_score(y_test, y_pred, pos_label=1, zero_division=0))
        rec_approved = float(recall_score(y_test, y_pred, pos_label=1, zero_division=0))
        f1_approved = float(f1_score(y_test, y_pred, pos_label=1, zero_division=0))
        
        prec_rejected = float(precision_score(y_test, y_pred, pos_label=0, zero_division=0))
        rec_rejected = float(recall_score(y_test, y_pred, pos_label=0, zero_division=0))
        f1_rejected = float(f1_score(y_test, y_pred, pos_label=0, zero_division=0))
        
        macro_f1 = float(f1_score(y_test, y_pred, average='macro'))
        weighted_f1 = float(f1_score(y_test, y_pred, average='weighted'))
        
        cm = confusion_matrix(y_test, y_pred).tolist()
        # cm is [[TN, FP], [FN, TP]] where 0=Rejected, 1=Approved
        tn, fp, fn, tp = int(cm[0][0]), int(cm[0][1]), int(cm[1][0]), int(cm[1][1])
        
        clf_report_dict = classification_report(
            y_test, y_pred,
            target_names=['N (Rejected)', 'Y (Approved)'],
            output_dict=True
        )
        
        test_results[name] = {
            'accuracy': round(acc, 4),
            'precision_approved': round(prec_approved, 4),
            'recall_approved': round(rec_approved, 4),
            'f1_approved': round(f1_approved, 4),
            'precision_rejected': round(prec_rejected, 4),
            'recall_rejected': round(rec_rejected, 4),
            'f1_rejected': round(f1_rejected, 4),
            'macro_f1': round(macro_f1, 4),
            'weighted_f1': round(weighted_f1, 4),
            'confusion_matrix': {
                'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
                'matrix': cm
            },
            'classification_report': clf_report_dict
        }
        
        print(f"      === {name} (Held-out Test Evaluation) ===")
        print(f"      Accuracy:             {acc*100:.2f}%")
        print(f"      Approved F1 (Class Y): {f1_approved:.4f} (Prec: {prec_approved:.4f}, Rec: {rec_approved:.4f})")
        print(f"      Rejected F1 (Class N): {f1_rejected:.4f} (Prec: {prec_rejected:.4f}, Rec: {rec_rejected:.4f})")
        print(f"      Confusion Matrix:     TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    
    return test_results

# ------------------------------------------------------------------------------
# 7. COMPARISON VISUALIZATION (MATPLOTLIB)
# ------------------------------------------------------------------------------
def generate_comparison_chart(cv_results: dict, test_results: dict, selected_model: str, output_path: str):
    """
    Renders a publication-quality comparison bar chart showing:
      Subplot 1: Held-out Test Set Performance Metrics across the 3 algorithms
      Subplot 2: 5-Fold Cross-Validation Metrics on Training Set with error indicators
    Saves image to static/comparison.png.
    """
    print(f"[6/7] Generating comparative performance visualization at: {output_path}")
    models = list(cv_results.keys())
    metrics = ['Accuracy', 'Precision (Y)', 'Recall (Y)', 'F1-score (Y)']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor('#ffffff')
    
    # Custom color palette
    colors = ['#1d4ed8', '#0284c7', '#10b981'] # Blue, Sky, Emerald
    x = np.arange(len(metrics))
    width = 0.25
    
    # --- Subplot 1: Held-out Test Set Metrics ---
    for idx, model_name in enumerate(models):
        res = test_results[model_name]
        vals = [res['accuracy'], res['precision_approved'], res['recall_approved'], res['f1_approved']]
        is_selected = (model_name == selected_model)
        label = f"{model_name} [Best Model]" if is_selected else model_name
        
        bars = ax1.bar(x + (idx - 1) * width, vals, width, label=label,
                       color=colors[idx], edgecolor='#0f172a', linewidth=1.2 if is_selected else 0.5, alpha=0.9)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f"{height:.2f}",
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3), textcoords="offset points",
                         ha='center', va='bottom', fontsize=9, fontweight='bold' if is_selected else 'normal')
            
    ax1.set_title("Held-out Test Set Performance (N=154)", fontsize=13, fontweight='bold', pad=12, color='#1e293b')
    ax1.set_ylabel("Score (0.0 to 1.0)", fontsize=11, fontweight='semibold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, fontsize=10, fontweight='semibold')
    ax1.set_ylim(0, 1.15)
    ax1.grid(axis='y', linestyle='--', alpha=0.4)
    ax1.legend(loc='lower right', frameon=True, facecolor='#f8fafc', edgecolor='#cbd5e1')
    
    # --- Subplot 2: Cross-Validation Metrics on Training Set ---
    for idx, model_name in enumerate(models):
        res = cv_results[model_name]
        vals = [res['accuracy_mean'], res['precision_mean'], res['recall_mean'], res['f1_mean']]
        is_selected = (model_name == selected_model)
        label = f"{model_name} [Selected]" if is_selected else model_name
        
        bars = ax2.bar(x + (idx - 1) * width, vals, width, label=label,
                       color=colors[idx], edgecolor='#0f172a', linewidth=1.2 if is_selected else 0.5, alpha=0.9)
        
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f"{height:.2f}",
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3), textcoords="offset points",
                         ha='center', va='bottom', fontsize=9, fontweight='bold' if is_selected else 'normal')
            
    ax2.set_title("5-Fold Stratified Cross-Validation on Training Set (N=460)", fontsize=13, fontweight='bold', pad=12, color='#1e293b')
    ax2.set_ylabel("Mean CV Score (0.0 to 1.0)", fontsize=11, fontweight='semibold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics, fontsize=10, fontweight='semibold')
    ax2.set_ylim(0, 1.15)
    ax2.grid(axis='y', linestyle='--', alpha=0.4)
    ax2.legend(loc='lower right', frameon=True, facecolor='#f8fafc', edgecolor='#cbd5e1')
    
    plt.suptitle("Comparative Analysis of Machine Learning Classification Algorithms for Loan Approval Prediction",
                 fontsize=14, fontweight='bold', y=0.98, color='#0f172a')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"      Chart successfully saved to {output_path}")

# ------------------------------------------------------------------------------
# 8. ARTIFACT PERSISTENCE (JOBLIB & JSON)
# ------------------------------------------------------------------------------
def save_artifacts(pipelines: dict, selected_model: str, cv_results: dict,
                   test_results: dict, dataset_stats: dict):
    """
    Serializes:
      1. models/loan_models.joblib: full pipeline objects, expected schema, categorical options.
      2. models/metrics.json: complete statistical and diagnostic evaluation metadata.
    """
    print("[7/7] Persisting model bundle and evaluation metrics...")
    
    categorical_options = {
        'Gender': ['Male', 'Female'],
        'Married': ['Yes', 'No'],
        'Dependents': ['0', '1', '2', '3'],
        'Education': ['Graduate', 'Not Graduate'],
        'Self_Employed': ['No', 'Yes'],
        'Property_Area': ['Urban', 'Semiurban', 'Rural']
    }
    
    model_bundle = {
        'models': pipelines,
        'selected_model': selected_model,
        'feature_names': NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        'numeric_features': NUMERIC_FEATURES,
        'categorical_features': CATEGORICAL_FEATURES,
        'categorical_options': categorical_options,
        'target_mapping': {1: 'Approved (Y)', 0: 'Rejected (N)'},
        'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    joblib.dump(model_bundle, MODEL_BUNDLE_PATH)
    print(f"      Saved model bundle to: {MODEL_BUNDLE_PATH}")
    
    metrics_payload = {
        'project_title': "Comparative Analysis of Machine Learning Algorithms for Loan Approval Prediction Using Python",
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dataset_statistics': dataset_stats,
        'model_selection': {
            'selected_model': selected_model,
            'selection_criterion': "Highest mean 5-fold Stratified Cross-Validation F1-score on the training dataset",
            'tie_breaking_rule': "Higher Cross-Validation Accuracy"
        },
        'cross_validation_results': cv_results,
        'held_out_test_results': test_results
    }
    
    with open(METRICS_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(metrics_payload, f, indent=4)
    print(f"      Saved comprehensive metrics JSON to: {METRICS_JSON_PATH}")

# ------------------------------------------------------------------------------
# 9. MAIN EXECUTION ENTRY POINT
# ------------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("LOAN APPROVAL PREDICTION - MACHINE LEARNING PIPELINE & MODEL COMPARISON")
    print("=" * 80)
    try:
        # Load & validate
        df = load_and_validate_dataset(DATASET_PATH)
        dataset_stats = {
            'total_samples': int(len(df)),
            'columns': list(df.columns),
            'target_distribution': {
                'Approved_Y': int((df[TARGET_COLUMN] == 'Y').sum()),
                'Rejected_N': int((df[TARGET_COLUMN] == 'N').sum())
            }
        }
        
        # Preprocess & split
        X_train, X_test, y_train, y_test = prepare_data(df)
        dataset_stats['train_samples'] = int(len(X_train))
        dataset_stats['test_samples'] = int(len(X_test))
        
        # Build pipelines
        pipelines = create_model_pipelines()
        
        # Cross-validation & selection strictly on training data
        cv_results, selected_model = perform_model_selection(pipelines, X_train, y_train)
        
        # Evaluate on held-out test data
        test_results = evaluate_on_held_out_test(pipelines, X_train, y_train, X_test, y_test)
        
        # Visualization
        generate_comparison_chart(cv_results, test_results, selected_model, COMPARISON_CHART_PATH)
        
        # Persistence
        save_artifacts(pipelines, selected_model, cv_results, test_results, dataset_stats)
        
        print("\n" + "=" * 80)
        print(" TRAINING & EVALUATION PIPELINE COMPLETED SUCCESSFULLY!")
        print(f" Selected Model: {selected_model}")
        print(" Artifacts generated:")
        print(f"   - Models:        {MODEL_BUNDLE_PATH}")
        print(f"   - Metrics:       {METRICS_JSON_PATH}")
        print(f"   - Visualization: {COMPARISON_CHART_PATH}")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n[FATAL ERROR]: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
