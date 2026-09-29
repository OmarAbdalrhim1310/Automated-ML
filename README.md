# 🤖 Autonomous AI Data Scientist & AutoML

An autonomous **Data Science and Automated Machine Learning platform** built with Python and Streamlit.

The application allows users to upload a CSV or Excel dataset, automatically inspect its structure and data quality, analyze the target variable, run an AutoML pipeline, compare multiple machine learning models, and interact with the dataset through a Groq-powered AI Data Science Agent.

The project is designed to make the complete machine learning workflow easier, more reproducible, and less dependent on manual model selection.

---

## 🚀 Features

### 📂 Dataset Upload

Upload datasets directly through the Streamlit interface.

Supported formats:

* CSV
* Excel (`.xlsx`)

The application automatically analyzes the uploaded dataset before training.

---

### 🔍 Automatic Dataset Profiling

The Data Science Agent automatically inspects:

* Number of rows and columns
* Numerical columns
* Categorical columns
* Data types
* Missing values
* Duplicate rows
* Unique values
* Constant columns

This information is used as evidence for the subsequent machine learning workflow.

---

### 🎯 Automatic Target Analysis

The application can analyze a selected target and determine whether the problem is:

* Classification
* Regression
* Unsupervised clustering

For classification problems, it also analyzes class balance and determines whether the dataset is balanced or imbalanced.

---

### 🧹 Data Quality & Preprocessing

The project includes an automatic preprocessing module that handles:

* Missing numerical values using the mean
* Missing categorical values using the mode
* Numerical outliers using statistical clipping
* Data preparation for machine learning models

The preprocessing module identifies extreme numerical values using a simplified Z-score approach and clips values beyond three standard deviations.

The AutoML engine also contains a leakage-safe preprocessing pipeline with numerical imputation, categorical imputation, and one-hot encoding.

---

### 🧠 Autonomous AutoML Engine

The AutoML engine evaluates multiple machine learning algorithms instead of relying on a single model.

The project includes models such as:

* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting
* Support Vector Machines
* Ridge Regression
* MLP Neural Networks
* XGBoost
* LightGBM
* CatBoost

Optional boosting frameworks are detected automatically when installed.

---

### ⚙️ Hyperparameter Tuning

The AutoML engine supports automated hyperparameter search using:

* `RandomizedSearchCV`
* Cross-validation
* Stratified cross-validation for classification
* K-Fold validation for regression

The system compares model results and produces a leaderboard containing the evaluated models and their scores.

---

### ⚖️ Class Imbalance Handling

For classification datasets, the engine analyzes the class distribution.

Supported imbalance strategies include:

* No Treatment
* Class Weighting
* SMOTE
* Random Oversampling
* Random Undersampling

These techniques are integrated into the training pipeline rather than being applied incorrectly before model validation.

---

### 🔒 Leakage-Safe Machine Learning Pipeline

The project is designed around a leakage-safe preprocessing and training workflow.

Preprocessing operations such as feature transformation and sampling are integrated into the machine learning pipeline so that transformations can be fitted within the appropriate training process.

The application also uses a deterministic AutoML policy so that the same dataset and target produce the same configuration and reproducible behavior.

---

### 💾 Experiment Persistence

AutoML experiments are stored using SQLite.

The experiment database records information such as:

* Task type
* Target column
* Winning model
* Score
* Training mode
* Imbalance method
* Feature engineering configuration
* Outlier configuration
* Experiment ID
* Complete result data

A stable experiment ID is generated from the dataset, target, and AutoML configuration.

This allows previous experiment results to be cached and retrieved.

---

### 📊 Interactive Dashboard

The Streamlit interface provides interactive data analysis including:

* Dataset overview
* Dataset shape
* Raw dataset preview
* Interactive visualizations
* Distribution analysis
* Model leaderboard
* AutoML results
* Data quality reports

Plotly is optionally used for interactive visualizations.

---

### 🤖 AI Data Science Agent

The project includes a Groq-powered Data Science Agent.

The agent can use local Python tools to:

1. Profile the dataset
2. Analyze the target
3. Analyze data quality
4. Run preprocessing audits
5. Execute the AutoML engine
6. Inspect the resulting metrics
7. Generate a grounded data science report

The LLM does not directly train models or invent results. The Python application executes the local analysis tools and sends the actual results back to the agent.

---

## 🏗️ Architecture

```text
                    ┌─────────────────────┐
                    │   Streamlit App     │
                    │      app.py         │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Dataset Upload    │
                    │    CSV / Excel      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Dataset Profiling   │
                    │ Target Analysis     │
                    │ Data Quality        │
                    └──────────┬──────────┘
                               │
                 ┌─────────────┴─────────────┐
                 ▼                           ▼
        ┌─────────────────┐        ┌─────────────────┐
        │  Preprocessing  │        │  Groq AI Agent  │
        │ preprocessing.py│        │  ds_agent.py    │
        └────────┬────────┘        └────────┬────────┘
                 │                           │
                 └─────────────┬─────────────┘
                               ▼
                    ┌─────────────────────┐
                    │    AutoML Engine    │
                    │ automl_engine.py    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Model Training &     │
                    │ Hyperparameter      │
                    │ Tuning              │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Model Leaderboard   │
                    │ & Evaluation        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ SQLite Experiments  │
                    └─────────────────────┘
```

---

## 📁 Project Structure

```text
Automated-ML/
│
├── app.py
│   └── Streamlit user interface
│
├── ds_agent.py
│   └── Groq-powered autonomous Data Science Agent
│
├── automl_engine.py
│   └── AutoML models, preprocessing pipelines,
│       hyperparameter tuning and experiment persistence
│
├── preprocessing.py
│   └── Data cleaning and preprocessing utilities
│
├── README.md
│   └── Project documentation
│
└── automl_experiments.db
    └── SQLite database generated by the application
```

The core application modules are separated into the Streamlit interface, Data Science Agent, AutoML engine, and preprocessing module.

---

## 🛠️ Technologies

### Programming Language

* Python

### Data Science

* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* Imbalanced-learn

### Machine Learning Models

* Logistic Regression
* Ridge Regression
* Decision Trees
* Random Forest
* Gradient Boosting
* SVM
* MLP Neural Networks
* XGBoost
* LightGBM
* CatBoost

### Visualization

* Plotly
* Streamlit

### AI

* Groq API
* Groq-powered Data Science Agent

### Storage

* SQLite

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/OmarAbdalrhim1310/Automated-ML.git
cd Automated-ML
```

### 2. Create a virtual environment

On Windows:

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available yet, the main project dependencies include packages such as:

```bash
pip install streamlit pandas numpy scikit-learn imbalanced-learn plotly groq
```

Optional ML libraries used by the AutoML engine:

```bash
pip install xgboost lightgbm catboost
```

---

## 🔑 Groq API Key

The conversational AI functionality requires a Groq API key.

The application initializes the Groq client using the supplied API key and uses the configured model for the Data Science Agent.

Set your API key as an environment variable:

### Windows PowerShell

```powershell
$env:GROQ_API_KEY="your_api_key"
```

Or configure the key through the application's Streamlit interface if your current `app.py` provides that option.

**Never commit your API key to GitHub.**

---

## ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

Then open the local URL shown by Streamlit in your browser.

---

## 🔄 Typical Workflow

```text
1. Upload Dataset
        ↓
2. Inspect Dataset
        ↓
3. Select Target
        ↓
4. Analyze Data Quality
        ↓
5. Determine ML Task
        ↓
6. Analyze Class Balance
        ↓
7. Run AutoML
        ↓
8. Train Multiple Models
        ↓
9. Hyperparameter Tuning
        ↓
10. Compare Models
        ↓
11. Select Model
        ↓
12. Generate Results
        ↓
13. Ask the AI Data Scientist
```

The Data Science Agent follows an evidence-based workflow: it inspects the dataset and target first, checks data quality and class imbalance, and then runs AutoML when appropriate.

---

## 📈 Evaluation Metrics

### Classification

The AutoML engine includes metrics such as:

* Accuracy
* Balanced Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

### Regression

The engine includes:

* MAE
* MSE
* RMSE
* R²

### Clustering

The project includes:

* Silhouette Score

The metrics are implemented using Scikit-learn evaluation functions in the AutoML engine.

---

## 🔁 Reproducibility

The AutoML engine uses a fixed random state:

```python
RANDOM_STATE = 42
```

The project also generates a stable experiment fingerprint based on the dataset, target, column information, data types, and AutoML configuration.

This makes repeated experiments more reproducible.

---

## 🧪 Example Use Cases

This platform can be used for:

* Exploratory dataset analysis
* Classification problems
* Regression problems
* Unsupervised clustering
* Automated model comparison
* Hyperparameter optimization
* Data quality auditing
* Class imbalance analysis
* Interactive machine learning dashboards
* Conversational data analysis

---

## ⚠️ Important Notes

* The quality of AutoML results depends on the quality and size of the uploaded dataset.
* Optional models such as XGBoost, LightGBM, and CatBoost are used only when their packages are available.
* The Groq API is required for the conversational AI functionality.
* Experiment history is stored locally in SQLite.
* API keys and other secrets should never be committed to the repository.

---

## 👨‍💻 Author

**Omar Abdalrhim**

GitHub:
https://github.com/OmarAbdalrhim1310

---

## 📄 License

This project is intended for educational and machine learning development purposes.
