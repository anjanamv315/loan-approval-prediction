"""
================================================================================
Comparative Analysis of Machine Learning Algorithms for Loan Approval Prediction
File: app.py
Purpose: Production-grade, educational Flask web application, SQLite integration,
         dynamic prediction interface, and model comparison dashboard.
Author: Academic MTech AI & Data Science Project
================================================================================
"""

import os
import json
import sqlite3
import joblib
from datetime import datetime
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash

# ------------------------------------------------------------------------------
# 1. APPLICATION CONFIGURATION & PATH SETUP
# ------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
STATIC_DIR = os.path.join(BASE_DIR, 'static')
MODEL_BUNDLE_PATH = os.path.join(MODELS_DIR, 'loan_models.joblib')
METRICS_JSON_PATH = os.path.join(MODELS_DIR, 'metrics.json')
DATABASE_PATH = os.path.join(BASE_DIR, 'loan_history.db')

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'mtech_loan_prediction_academic_secret_key_42')

# ------------------------------------------------------------------------------
# 2. DATABASE INITIALIZATION & UTILITIES
# ------------------------------------------------------------------------------
def get_db_connection():
    """Returns a SQLite connection with dict-like row access."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Creates the predictions table automatically if it does not already exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            gender TEXT NOT NULL,
            married TEXT NOT NULL,
            dependents TEXT NOT NULL,
            education TEXT NOT NULL,
            self_employed TEXT NOT NULL,
            applicant_income REAL NOT NULL,
            coapplicant_income REAL NOT NULL,
            loan_amount REAL NOT NULL,
            loan_amount_term REAL NOT NULL,
            credit_history REAL NOT NULL,
            property_area TEXT NOT NULL,
            lr_prediction TEXT NOT NULL,
            lr_probability REAL NOT NULL,
            dt_prediction TEXT NOT NULL,
            dt_probability REAL NOT NULL,
            rf_prediction TEXT NOT NULL,
            rf_probability REAL NOT NULL,
            selected_model_name TEXT NOT NULL,
            selected_model_prediction TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

# Initialize database on startup
init_db()

# ------------------------------------------------------------------------------
# 3. MODEL & METRICS LOADING UTILITIES
# ------------------------------------------------------------------------------
def load_model_bundle():
    """
    Safely loads serialized pipelines and metadata from joblib.
    Returns (bundle, None) on success or (None, error_message) on failure.
    """
    if not os.path.exists(MODEL_BUNDLE_PATH):
        return None, "Model bundle not found. Please run 'python train_model.py' to generate trained models."
    try:
        bundle = joblib.load(MODEL_BUNDLE_PATH)
        return bundle, None
    except Exception as e:
        return None, f"Failed to load trained model bundle: {str(e)}"

def load_metrics_json():
    """
    Safely loads comparative metrics from JSON.
    Returns (metrics_dict, None) on success or (None, error_message) on failure.
    """
    if not os.path.exists(METRICS_JSON_PATH):
        return None, "Evaluation metrics file not found. Please run 'python train_model.py'."
    try:
        with open(METRICS_JSON_PATH, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
        return metrics, None
    except Exception as e:
        return None, f"Failed to load evaluation metrics: {str(e)}"

# ------------------------------------------------------------------------------
# 4. SERVER-SIDE INPUT VALIDATION & SANITIZATION
# ------------------------------------------------------------------------------
VALID_CATEGORIES = {
    'Gender': ['Male', 'Female'],
    'Married': ['Yes', 'No'],
    'Dependents': ['0', '1', '2', '3+'],
    'Education': ['Graduate', 'Not Graduate'],
    'Self_Employed': ['No', 'Yes'],
    'Property_Area': ['Urban', 'Semiurban', 'Rural']
}

def validate_and_parse_input(form_data):
    """
    Validates form inputs for missing values, allowable categories, and realistic numeric ranges.
    Returns (clean_dict, None) on success or (None, error_list) on validation failure.
    """
    errors = []
    clean = {}

    # 1. Categorical fields
    for field, valid_options in VALID_CATEGORIES.items():
        val = form_data.get(field, '').strip()
        if not val:
            errors.append(f"Please select an option for '{field}'.")
        elif val not in valid_options:
            errors.append(f"Invalid option '{val}' for '{field}'. Allowed: {', '.join(valid_options)}.")
        else:
            clean[field] = val

    # 2. Numerical fields
    # Applicant Income
    app_inc_raw = form_data.get('ApplicantIncome', '').strip()
    try:
        app_inc = float(app_inc_raw)
        if app_inc < 0:
            errors.append("Applicant Income must be a non-negative number.")
        clean['ApplicantIncome'] = app_inc
    except ValueError:
        errors.append("Applicant Income must be a valid number.")

    # Coapplicant Income
    coapp_inc_raw = form_data.get('CoapplicantIncome', '').strip()
    try:
        coapp_inc = float(coapp_inc_raw)
        if coapp_inc < 0:
            errors.append("Co-applicant Income must be a non-negative number.")
        clean['CoapplicantIncome'] = coapp_inc
    except ValueError:
        errors.append("Co-applicant Income must be a valid number.")

    # Loan Amount (in thousands)
    loan_amt_raw = form_data.get('LoanAmount', '').strip()
    try:
        loan_amt = float(loan_amt_raw)
        if loan_amt <= 0:
            errors.append("Loan Amount must be greater than 0.")
        clean['LoanAmount'] = loan_amt
    except ValueError:
        errors.append("Loan Amount must be a valid positive number.")

    # Loan Amount Term (in months)
    term_raw = form_data.get('Loan_Amount_Term', '').strip()
    try:
        term = float(term_raw)
        if term <= 0:
            errors.append("Loan Amount Term must be a positive number of months (e.g., 360).")
        clean['Loan_Amount_Term'] = term
    except ValueError:
        errors.append("Loan Amount Term must be a valid number.")

    # Credit History (1.0 or 0.0)
    ch_raw = form_data.get('Credit_History', '').strip()
    try:
        ch = float(ch_raw)
        if ch not in [0.0, 1.0]:
            errors.append("Credit History must be 1 (Meets Guidelines) or 0 (Does Not Meet Guidelines).")
        clean['Credit_History'] = ch
    except ValueError:
        errors.append("Please select a valid Credit History option.")

    if errors:
        return None, errors
    return clean, None

# ------------------------------------------------------------------------------
# 5. WEB APPLICATION ROUTES
# ------------------------------------------------------------------------------
@app.route('/')
def home():
    """Landing page showcasing project overview, ML methodology, and quick actions."""
    metrics, _ = load_metrics_json()
    model_bundle, _ = load_model_bundle()
    models_ready = (model_bundle is not None and metrics is not None)
    selected_model_name = metrics['model_selection']['selected_model'] if models_ready else "Random Forest"
    return render_template('index.html',
                           models_ready=models_ready,
                           selected_model_name=selected_model_name,
                           metrics=metrics)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Loan application submission form and multi-model inference handler."""
    model_bundle, err = load_model_bundle()
    if err:
        flash(err, 'danger')
        return render_template('predict.html', model_bundle=None)

    if request.method == 'GET':
        return render_template('predict.html', model_bundle=model_bundle, form_values={})

    # Process POST request
    clean_input, validation_errors = validate_and_parse_input(request.form)
    if validation_errors:
        for error_msg in validation_errors:
            flash(error_msg, 'danger')
        return render_template('predict.html', model_bundle=model_bundle, form_values=request.form)

    try:
        # Preprocess input data to match training pipeline schema exactly
        # Note: '3+' was mapped to '3' during training
        dependents_clean = '3' if clean_input['Dependents'] == '3+' else clean_input['Dependents']

        input_dict = {
            'ApplicantIncome': [clean_input['ApplicantIncome']],
            'CoapplicantIncome': [clean_input['CoapplicantIncome']],
            'LoanAmount': [clean_input['LoanAmount']],
            'Loan_Amount_Term': [clean_input['Loan_Amount_Term']],
            'Credit_History': [clean_input['Credit_History']],
            'Gender': [clean_input['Gender']],
            'Married': [clean_input['Married']],
            'Dependents': [dependents_clean],
            'Education': [clean_input['Education']],
            'Self_Employed': [clean_input['Self_Employed']],
            'Property_Area': [clean_input['Property_Area']]
        }

        # Build DataFrame with exact feature order expected by pipelines
        df_inference = pd.DataFrame(input_dict)

        pipelines = model_bundle['models']
        selected_model_name = model_bundle['selected_model']

        predictions = {}
        probabilities = {}

        # Generate predictions for all three algorithms
        for name in ['Logistic Regression', 'Decision Tree', 'Random Forest']:
            pipe = pipelines[name]
            pred_class = int(pipe.predict(df_inference)[0])
            predictions[name] = 'Approved' if pred_class == 1 else 'Rejected'

            # Calculate probability of approval (class 1)
            if hasattr(pipe, 'predict_proba'):
                prob_arr = pipe.predict_proba(df_inference)[0]
                # Class 1 is approval
                approval_prob = round(float(prob_arr[1]) * 100, 2)
            else:
                approval_prob = 100.0 if pred_class == 1 else 0.0
            probabilities[name] = approval_prob

        selected_pred = predictions[selected_model_name]

        # Save to SQLite database
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO predictions (
                timestamp, gender, married, dependents, education, self_employed,
                applicant_income, coapplicant_income, loan_amount, loan_amount_term,
                credit_history, property_area,
                lr_prediction, lr_probability,
                dt_prediction, dt_probability,
                rf_prediction, rf_probability,
                selected_model_name, selected_model_prediction
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            now_str,
            clean_input['Gender'],
            clean_input['Married'],
            clean_input['Dependents'],
            clean_input['Education'],
            clean_input['Self_Employed'],
            clean_input['ApplicantIncome'],
            clean_input['CoapplicantIncome'],
            clean_input['LoanAmount'],
            clean_input['Loan_Amount_Term'],
            clean_input['Credit_History'],
            clean_input['Property_Area'],
            predictions['Logistic Regression'],
            probabilities['Logistic Regression'],
            predictions['Decision Tree'],
            probabilities['Decision Tree'],
            predictions['Random Forest'],
            probabilities['Random Forest'],
            selected_model_name,
            selected_pred
        ))
        record_id = cursor.lastrowid
        conn.commit()
        conn.close()

        flash("Loan prediction successfully computed and archived in the database.", "success")
        return render_template(
            'result.html',
            record_id=record_id,
            timestamp=now_str,
            applicant_data=clean_input,
            predictions=predictions,
            probabilities=probabilities,
            selected_model_name=selected_model_name,
            selected_model_prediction=selected_pred
        )

    except Exception as e:
        flash(f"An error occurred while generating predictions: {str(e)}", "danger")
        return render_template('predict.html', model_bundle=model_bundle, form_values=request.form)

@app.route('/comparison')
def comparison():
    """Model comparison dashboard presenting CV scores, held-out evaluation, and Matplotlib chart."""
    metrics, err = load_metrics_json()
    if err:
        flash(err, 'danger')
        return render_template('comparison.html', metrics=None)

    chart_exists = os.path.exists(os.path.join(STATIC_DIR, 'comparison.png'))
    return render_template('comparison.html', metrics=metrics, chart_exists=chart_exists)

@app.route('/history')
def history():
    """Prediction history page retrieving logged user applications in reverse chronological order."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions ORDER BY id DESC")
    records = cursor.fetchall()
    conn.close()
    return render_template('history.html', records=records)

# ------------------------------------------------------------------------------
# 6. APPLICATION ENTRY POINT
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 70)
    print(" LOAN APPROVAL PREDICTION FLASK WEB SERVER")
    print(" Local URL: http://127.0.0.1:5000")
    print(" Academic project: Strictly for educational analysis.")
    print("=" * 70)
    app.run(host='127.0.0.1', port=5000, debug=True)
