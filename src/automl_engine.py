# ============================================================
# AUTONOMOUS AUTOML ENGINE
# Stage 1 - Clean, Leakage-Safe, Imbalance-Aware AutoML
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import os
import sqlite3
import hashlib
import json
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectPercentile, mutual_info_classif, mutual_info_regression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler

# Optional Modern Boosters
try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    HAS_CATBOOST = True
except Exception:
    HAS_CATBOOST = False

RANDOM_STATE = 42
os.environ.setdefault("PYTHONHASHSEED", str(RANDOM_STATE))
np.random.seed(RANDOM_STATE)
DB_PATH = "automl_experiments.db"

# ============================================================
# 0. EXPERIMENT PERSISTENCE (SQLITE)
# ============================================================

def init_experiments_db():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                task TEXT,
                target TEXT,
                winner TEXT,
                score REAL,
                training_mode TEXT,
                imbalance_method TEXT,
                feature_engineering INTEGER,
                outliers TEXT,
                experiment_id TEXT,
                result_json TEXT
            )
            """
        )
        # Backward-compatible migration for existing databases.
        for col, ddl in [("experiment_id", "TEXT"), ("result_json", "TEXT")]:
            try:
                cursor.execute(f"ALTER TABLE experiments ADD COLUMN {col} {ddl}")
            except sqlite3.OperationalError:
                pass
        conn.commit()
        conn.close()
    except Exception:
        pass

def log_experiment_to_db(task, target, winner, score, training_mode, imbalance_method, feature_engineering, outliers, experiment_id=None, result_json=None):
    try:
        init_experiments_db()
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        val = float(str(score).replace("%", ""))
        cursor.execute(
            """
            INSERT INTO experiments (task, target, winner, score, training_mode, imbalance_method, feature_engineering, outliers, experiment_id, result_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (task, target, winner, val, training_mode, imbalance_method, int(feature_engineering), outliers, experiment_id, result_json)
        )
        conn.commit()
        conn.close()
    except Exception:
        pass

def make_experiment_id(df, target_column, **config):
    """Stable fingerprint: same data + target + AutoML policy => same experiment id."""
    hasher = hashlib.sha256()
    # pandas hashing is row/order sensitive and deterministic for the same frame.
    frame = df.copy()
    hasher.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    hasher.update(json.dumps({
        "columns": [str(c) for c in frame.columns],
        "dtypes": [str(t) for t in frame.dtypes],
        "target": target_column,
        **config,
    }, sort_keys=True, default=str).encode("utf-8"))
    return hasher.hexdigest()[:16]

def fetch_cached_experiment(experiment_id):
    init_experiments_db()
    try:
        conn = sqlite3.connect(DB_PATH)
        row = pd.read_sql_query(
            "SELECT result_json FROM experiments WHERE experiment_id = ? AND result_json IS NOT NULL ORDER BY id DESC LIMIT 1",
            conn, params=(experiment_id,)
        )
        conn.close()
        if row.empty or not row.iloc[0]["result_json"]:
            return None
        return json.loads(row.iloc[0]["result_json"])
    except Exception:
        return None


def fetch_experiments_history():
    init_experiments_db()
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM experiments ORDER BY id DESC", conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()

# ============================================================
# 1. BASIC DATA TYPE DETECTION
# ============================================================

def detect_data_type(df):
    if df is None or df.empty:
        return "Unknown"

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns

    if len(numeric_cols) > 0 and len(non_numeric_cols) == 0:
        return "Regression / Numeric Analysis"

    if len(non_numeric_cols) > 0 and len(numeric_cols) == 0:
        return "Classification / Categorical Analysis"

    return "Mixed Data / Feature Engineering"

# ============================================================
# 2. TARGET TYPE DETECTION
# ============================================================

def detect_classification_target(y):
    if y is None:
        return False

    y = pd.Series(y).dropna()

    if y.empty or y.nunique() < 2:
        return False

    if (
        pd.api.types.is_object_dtype(y)
        or isinstance(y.dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(y)
        or pd.api.types.is_string_dtype(y)
    ):
        return True

    if pd.api.types.is_integer_dtype(y):
        unique_values = y.nunique()
        unique_ratio = unique_values / max(len(y), 1)

        if unique_values <= 20 and unique_ratio <= 0.10:
            return True

    return False

# ============================================================
# 3. CLASS BALANCE ANALYSIS
# ============================================================

def analyze_class_balance(y):
    if y is None:
        return {
            "is_classification": False,
            "status": "Unknown",
            "ratio": None,
            "minority_percentage": None,
            "majority_percentage": None,
            "counts": pd.Series(dtype=int),
            "percentages": pd.Series(dtype=float),
        }

    y = pd.Series(y).dropna()

    if y.empty:
        return {
            "is_classification": False,
            "status": "Unknown",
            "ratio": None,
            "minority_percentage": None,
            "majority_percentage": None,
            "counts": pd.Series(dtype=int),
            "percentages": pd.Series(dtype=float),
        }

    if not detect_classification_target(y):
        return {
            "is_classification": False,
            "status": "Not Classification",
            "ratio": None,
            "minority_percentage": None,
            "majority_percentage": None,
            "counts": pd.Series(dtype=int),
            "percentages": pd.Series(dtype=float),
        }

    counts = y.value_counts(dropna=False)

    if len(counts) < 2:
        return {
            "is_classification": True,
            "status": "Invalid - One Class Only",
            "ratio": np.inf,
            "minority_percentage": 100.0,
            "majority_percentage": 100.0,
            "counts": counts,
            "percentages": pd.Series([100.0], index=counts.index),
        }

    majority = int(counts.max())
    minority = int(counts.min())
    ratio = majority / max(minority, 1)

    percentages = (counts / counts.sum() * 100).round(2)
    minority_percentage = float(percentages.min())
    majority_percentage = float(percentages.max())

    if ratio < 1.5:
        status = "Balanced"
    elif ratio < 2:
        status = "Mild Imbalance"
    elif ratio < 3:
        status = "Moderate Imbalance"
    else:
        status = "Severe Imbalance"

    return {
        "is_classification": True,
        "status": status,
        "ratio": float(ratio),
        "minority_percentage": minority_percentage,
        "majority_percentage": majority_percentage,
        "counts": counts,
        "percentages": percentages,
    }

class AutomaticFeatureEngineer(BaseEstimator, TransformerMixin):
    """Leakage-safe row-wise feature engineering for tabular and datetime data."""
    def __init__(self, enabled=True, max_numeric_features=12):
        self.enabled = enabled
        self.max_numeric_features = max_numeric_features

    def fit(self, X, y=None):
        X = X.copy()
        self.input_columns_ = list(X.columns)
        self.numeric_columns_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.numeric_columns_ = self.numeric_columns_[: self.max_numeric_features]
        self.datetime_columns_ = []
        for col in X.columns:
            if X[col].dtype == "object":
                try:
                    pd.to_datetime(X[col].dropna().head(10), errors="raise")
                    self.datetime_columns_.append(col)
                except Exception:
                    pass
        return self

    def transform(self, X):
        X = X.copy()
        if not self.enabled:
            return X
        # Datetime feature decomposition
        for col in getattr(self, "datetime_columns_", []):
            if col in X.columns:
                dt = pd.to_datetime(X[col], errors="coerce")
                X[f"{col}__year"] = dt.dt.year.fillna(0)
                X[f"{col}__month"] = dt.dt.month.fillna(0)
                X[f"{col}__day"] = dt.dt.day.fillna(0)
                X[f"{col}__dayofweek"] = dt.dt.dayofweek.fillna(0)
                X.drop(columns=[col], inplace=True)
        # Mathematical features
        for col in self.numeric_columns_:
            if col not in X.columns:
                continue
            values = pd.to_numeric(X[col], errors="coerce")
            safe = values.abs()
            X[f"{col}__sq"] = values ** 2
            X[f"{col}__log_abs"] = np.log1p(safe)
        return X

class QuantileOutlierHandler(BaseEstimator, TransformerMixin):
    """Training-fold-only winsorization for numeric columns."""
    def __init__(self, method="No Treatment", lower=0.01, upper=0.99):
        self.method = method
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = X.copy()
        self.numeric_columns_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}
        if self.method == "Winsorize":
            for col in self.numeric_columns_:
                series = pd.to_numeric(X[col], errors="coerce")
                self.lower_bounds_[col] = series.quantile(self.lower)
                self.upper_bounds_[col] = series.quantile(self.upper)
        return self

    def transform(self, X):
        X = X.copy()
        if self.method != "Winsorize":
            return X
        for col in self.numeric_columns_:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors="coerce").clip(
                    self.lower_bounds_[col], self.upper_bounds_[col]
                )
        return X

# ============================================================
# 4. PREPROCESSING
# ============================================================

def build_preprocessor(X):
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    transformers = []

    if numeric_features:
        numeric_pipeline = SklearnPipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )
        transformers.append(("num", numeric_pipeline, numeric_features))

    if categorical_features:
        categorical_pipeline = SklearnPipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]
        )
        transformers.append(("cat", categorical_pipeline, categorical_features))

    if not transformers:
        raise ValueError("No usable feature columns were found.")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

# ============================================================
# 5. IMBALANCE PIPELINE
# ============================================================

def build_imbalance_pipeline(
    preprocessor,
    model,
    imbalance_method="No Treatment",
    smote_k_neighbors=1,
    use_scaler=False,
    feature_engineering=False,
    outlier_method="No Treatment",
    feature_selection=False,
    feature_selection_percentile=80,
    selection_score_func=None,
):
    steps = [
        ("feature_engineer", AutomaticFeatureEngineer(enabled=feature_engineering)),
        ("outlier_handler", QuantileOutlierHandler(method=outlier_method)),
        ("preprocessor", preprocessor),
    ]
    if feature_selection:
        score_func = selection_score_func or mutual_info_regression
        steps.append(("feature_selector", SelectPercentile(score_func=score_func, percentile=feature_selection_percentile)))

    if imbalance_method == "SMOTE":
        steps.append(
            (
                "sampler",
                SMOTE(
                    random_state=RANDOM_STATE,
                    k_neighbors=smote_k_neighbors,
                ),
            )
        )
    elif imbalance_method == "Random Oversampling":
        steps.append(
            (
                "sampler",
                RandomOverSampler(random_state=RANDOM_STATE),
            )
        )
    elif imbalance_method == "Random Undersampling":
        steps.append(
            (
                "sampler",
                RandomUnderSampler(random_state=RANDOM_STATE),
            )
        )

    if use_scaler:
        steps.append(("scaler", StandardScaler()))

    steps.append(("model", model))

    if imbalance_method in {
        "SMOTE",
        "Random Oversampling",
        "Random Undersampling",
    }:
        return ImbPipeline(steps=steps)

    return SklearnPipeline(steps=steps)

# ============================================================
# 6. MODEL CONFIGURATION
# ============================================================

def get_classification_models(imbalance_method="No Treatment"):
    use_weights = imbalance_method == "Class Weighting"

    models = {
        "Logistic Regression": {
            "model": LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE,
                class_weight="balanced" if use_weights else None,
            ),
            "params": {
                "model__C": [0.1, 1.0, 10.0],
            },
            "scale": True,
            "supports_class_weight": True,
        },
        "Decision Tree": {
            "model": DecisionTreeClassifier(
                random_state=RANDOM_STATE,
                class_weight="balanced" if use_weights else None,
            ),
            "params": {
                "model__max_depth": [3, 5, 10, None],
                "model__min_samples_split": [2, 5],
            },
            "scale": False,
            "supports_class_weight": True,
        },
        "Random Forest": {
            "model": RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=1,
                class_weight="balanced" if use_weights else None,
            ),
            "params": {
                "model__n_estimators": [100, 200],
                "model__max_depth": [5, 10, None],
                "model__min_samples_split": [2, 5],
            },
            "scale": False,
            "supports_class_weight": True,
        },
        "Gradient Boosting": {
            "model": GradientBoostingClassifier(
                random_state=RANDOM_STATE,
            ),
            "params": {
                "model__n_estimators": [50, 100],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [2, 3],
            },
            "scale": False,
            "supports_class_weight": False,
        },
    }

    if HAS_XGB:
        models["XGBoost"] = {
            "model": XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss"),
            "params": {"model__n_estimators": [50, 100], "model__learning_rate": [0.05, 0.1]},
            "scale": False,
            "supports_class_weight": False,
        }
    if HAS_LGBM:
        models["LightGBM"] = {
            "model": LGBMClassifier(random_state=RANDOM_STATE, verbose=-1),
            "params": {"model__n_estimators": [50, 100], "model__learning_rate": [0.05, 0.1]},
            "scale": False,
            "supports_class_weight": False,
        }
    if HAS_CATBOOST:
        models["CatBoost"] = {
            "model": CatBoostClassifier(random_state=RANDOM_STATE, verbose=0),
            "params": {"model__iterations": [50, 100], "model__learning_rate": [0.05, 0.1]},
            "scale": False,
            "supports_class_weight": False,
        }

    models["SVM"] = {
        "model": SVC(
            random_state=RANDOM_STATE,
            class_weight="balanced" if use_weights else None,
            probability=True,
        ),
        "params": {
            "model__C": [0.1, 1.0, 10.0],
            "model__kernel": ["linear", "rbf"],
        },
        "scale": True,
        "supports_class_weight": True,
    }
    models["Neural Network"] = {
        "model": MLPClassifier(
            max_iter=300,
            random_state=RANDOM_STATE,
            early_stopping=True,
        ),
        "params": {
            "model__hidden_layer_sizes": [(50,), (50, 25)],
            "model__alpha": [0.0001, 0.001],
        },
        "scale": True,
        "supports_class_weight": False,
    }
    return models

def get_regression_models():
    models = {
        "Ridge Regression": {
            "model": Ridge(),
            "params": {
                "model__alpha": [0.1, 1.0, 10.0],
            },
            "scale": True,
        },
        "Decision Tree Regressor": {
            "model": DecisionTreeRegressor(
                random_state=RANDOM_STATE,
            ),
            "params": {
                "model__max_depth": [5, 10, None],
                "model__min_samples_split": [2, 5],
            },
            "scale": False,
        },
        "Random Forest Regressor": {
            "model": RandomForestRegressor(
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            "params": {
                "model__n_estimators": [100, 200],
                "model__max_depth": [5, 10, None],
            },
            "scale": False,
        },
        "Gradient Boosting Regressor": {
            "model": GradientBoostingRegressor(
                random_state=RANDOM_STATE,
            ),
            "params": {
                "model__n_estimators": [50, 100],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [2, 3],
            },
            "scale": False,
        },
    }

    if HAS_XGB:
        models["XGBoost Regressor"] = {
            "model": XGBRegressor(random_state=RANDOM_STATE),
            "params": {"model__n_estimators": [50, 100], "model__learning_rate": [0.05, 0.1]},
            "scale": False,
        }
    if HAS_LGBM:
        models["LightGBM Regressor"] = {
            "model": LGBMRegressor(random_state=RANDOM_STATE, verbose=-1),
            "params": {"model__n_estimators": [50, 100], "model__learning_rate": [0.05, 0.1]},
            "scale": False,
        }
    if HAS_CATBOOST:
        models["CatBoost Regressor"] = {
            "model": CatBoostRegressor(random_state=RANDOM_STATE, verbose=0),
            "params": {"model__iterations": [50, 100], "model__learning_rate": [0.05, 0.1]},
            "scale": False,
        }

    models["SVM Regressor"] = {
        "model": SVR(),
        "params": {
            "model__C": [0.1, 1.0, 10.0],
            "model__kernel": ["linear", "rbf"],
        },
        "scale": True,
    }
    models["Neural Network Regressor"] = {
        "model": MLPRegressor(
            max_iter=300,
            random_state=RANDOM_STATE,
            early_stopping=True,
        ),
        "params": {
            "model__hidden_layer_sizes": [(50,), (50, 25)],
            "model__alpha": [0.0001, 0.001],
        },
        "scale": True,
    }
    return models

# ============================================================
# 7. TRAINING MODES
# ============================================================

def get_training_configuration(training_mode="Fast"):
    cls_keys = list(get_classification_models().keys())
    reg_keys = list(get_regression_models().keys())

    configurations = {
        "Fast": {
            "cv": 2,
            "n_iter": 2,
            "classification_models": [m for m in ["Logistic Regression", "Decision Tree", "Random Forest", "LightGBM"] if m in cls_keys],
            "regression_models": [m for m in ["Ridge Regression", "Decision Tree Regressor", "Random Forest Regressor", "LightGBM Regressor"] if m in reg_keys],
        },
        "Balanced": {
            "cv": 3,
            "n_iter": 4,
            "classification_models": [m for m in ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting", "XGBoost", "LightGBM"] if m in cls_keys],
            "regression_models": [m for m in ["Ridge Regression", "Decision Tree Regressor", "Random Forest Regressor", "Gradient Boosting Regressor", "XGBoost Regressor", "LightGBM Regressor"] if m in reg_keys],
        },
        "Deep": {
            "cv": 3,
            "n_iter": 8,
            "classification_models": cls_keys,
            "regression_models": reg_keys,
        },
    }
    return configurations.get(training_mode, configurations["Fast"])

# ============================================================
# 8. HELPERS
# ============================================================

def calculate_total_combinations(param_grid):
    total = 1
    for values in param_grid.values():
        total *= len(values)
    return max(total, 1)

def explain_parameters(best_params):
    explanations = []
    for param, value in best_params.items():
        clean_param = param.replace("model__", "")
        if clean_param == "C": explanations.append(f"C={value}: controls inverse regularization strength.")
        elif clean_param == "kernel": explanations.append(f"kernel={value}: determines SVM decision-boundary shape.")
        elif clean_param == "max_depth": explanations.append(f"max_depth={value}: controls tree complexity and overfitting.")
        elif clean_param == "n_estimators": explanations.append(f"n_estimators={value}: number of trees/boosting stages.")
        elif clean_param == "learning_rate": explanations.append(f"learning_rate={value}: controls boosting step size.")
        elif clean_param == "hidden_layer_sizes": explanations.append(f"hidden_layer_sizes={value}: defines neural-network architecture.")
        elif clean_param == "alpha": explanations.append(f"alpha={value}: regularization strength.")
        elif clean_param == "min_samples_split": explanations.append(f"min_samples_split={value}: min samples to split node.")
    return " ".join(explanations) if explanations else "Default optimized configuration."

def _clean_features(X):
    X = X.copy()
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        X[numeric_cols] = X[numeric_cols].replace([np.inf, -np.inf], np.nan)
    all_missing = X.columns[X.isna().all()].tolist()
    if all_missing:
        X = X.drop(columns=all_missing)
    if X.shape[1] == 0:
        raise ValueError("No usable feature columns remain after cleaning.")
    return X

def _safe_classification_split(X, y):
    counts = pd.Series(y).value_counts()
    if len(counts) < 2:
        raise ValueError("Classification target must contain at least two classes.")
    if counts.min() < 2:
        raise ValueError("At least one class has fewer than 2 samples for stratified split.")
    return train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)

def _safe_cv_splits(y_train, requested_splits):
    min_class_count = int(pd.Series(y_train).value_counts().min())
    cv_splits = min(requested_splits, min_class_count)
    if cv_splits < 2:
        raise ValueError("Not enough minority-class samples for cross-validation.")
    return cv_splits

def _smote_settings(y_train, cv_splits):
    min_class_count = int(pd.Series(y_train).value_counts().min())
    held_out_per_fold = int(np.ceil(min_class_count / cv_splits))
    estimated_inner = min_class_count - held_out_per_fold
    if estimated_inner < 2:
        return False, 1
    return True, max(1, min(3, estimated_inner - 1))

# ============================================================
# 9. UNSUPERVISED CLUSTERING
# ============================================================

def _run_unsupervised(X, training_mode, logs):
    X = _clean_features(X)
    clustering_preprocessor = build_preprocessor(X)
    X_encoded = clustering_preprocessor.fit_transform(X)
    scaler = StandardScaler()
    X_encoded = scaler.fit_transform(X_encoded)

    if len(X_encoded) < 3:
        raise ValueError("At least 3 rows are required for clustering.")

    k_options = [2, 3, 4] if training_mode == "Fast" else [2, 3, 4, 5]
    results = []

    for k in k_options:
        if len(X_encoded) <= k:
            continue
        try:
            model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
            labels = model.fit_predict(X_encoded)
            score = silhouette_score(X_encoded, labels) * 100 if len(np.unique(labels)) > 1 else 0.0
            results.append({
                "Model Name": f"K-Means (k={k})",
                "Score": round(float(score), 2),
                "Metric": "Silhouette Score",
                "Best Parameters": f"n_clusters={k}",
            })
        except Exception as exc:
            logs.append(f"⚠️ K={k} failed: {exc}")

    if not results:
        raise RuntimeError("No clustering model could be trained.")

    leaderboard = pd.DataFrame(results).sort_values("Score", ascending=False).reset_index(drop=True)
    winner = leaderboard.iloc[0]["Model Name"]
    winner_score = float(leaderboard.iloc[0]["Score"])
    logs.append(f"🏆 Best clustering model: {winner}")
    logs.append(f"📈 Silhouette Score: {winner_score:.2f}%")

    return "Autonomous Unsupervised Clustering", leaderboard, winner, f"{winner_score:.2f}%", logs, {}, None, {}

# ============================================================
# 10. MODEL ARTIFACT / EXPLAINABILITY HELPERS
# ============================================================

def _get_selected_feature_names(fitted_pipeline):
    preprocessor = fitted_pipeline.named_steps.get("preprocessor")
    try:
        names = np.asarray(preprocessor.get_feature_names_out())
    except Exception:
        names = np.asarray([])
    selector = fitted_pipeline.named_steps.get("feature_selector")
    if selector is not None and hasattr(selector, "get_support") and len(names):
        names = names[selector.get_support()]
    return names

def _feature_importance_dataframe(fitted_pipeline, top_n=20):
    model = fitted_pipeline.named_steps.get("model")
    names = _get_selected_feature_names(fitted_pipeline)
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
        kind = "Feature Importance"
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        values = np.abs(coef) if coef.ndim == 1 else np.mean(np.abs(coef), axis=0)
        kind = "Absolute Coefficient"
    else:
        return pd.DataFrame(columns=["Feature", "Importance", "Type"])
    if len(names) != len(values):
        names = np.asarray([f"feature_{i}" for i in range(len(values))])
    return pd.DataFrame({"Feature": names, "Importance": values, "Type": kind}).sort_values("Importance", ascending=False).head(top_n).reset_index(drop=True)

def detect_target_leakage_risk(X, y, target_name):
    rows = []
    y_series = pd.Series(y).reset_index(drop=True)
    target_text = str(target_name).lower()
    for col in X.columns:
        name = str(col).lower()
        reasons, score = [], 0
        if target_text and target_text in name:
            reasons.append("feature name contains target name")
            score += 3
        series = X[col]
        if series.nunique(dropna=True) >= max(5, int(0.95 * len(series))) and series.nunique(dropna=True) == len(series.dropna()):
            if detect_classification_target(y_series):
                reasons.append("near-unique identifier-like feature")
                score += 2
        if pd.api.types.is_numeric_dtype(series) and pd.api.types.is_numeric_dtype(y_series):
            aligned = pd.DataFrame({"x": pd.to_numeric(series, errors="coerce"), "y": pd.to_numeric(y_series, errors="coerce")}).dropna()
            if len(aligned) >= 5 and aligned["x"].nunique() > 1 and aligned["y"].nunique() > 1:
                corr = abs(aligned["x"].corr(aligned["y"]))
                if pd.notna(corr) and corr >= 0.98:
                    reasons.append(f"very high target correlation ({corr:.3f})")
                    score += 4
        if score:
            rows.append({"Feature": col, "Risk Score": score, "Reason": "; ".join(reasons)})
    return pd.DataFrame(rows).sort_values("Risk Score", ascending=False).reset_index(drop=True) if rows else pd.DataFrame(columns=["Feature", "Risk Score", "Reason"])

def _classification_diagnostics(y_true, y_pred):
    labels = sorted(np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)])))
    return {
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0) * 100),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0) * 100),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
        "confusion_labels": [str(x) for x in labels]
    }

def _regression_diagnostics(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred)))
    }

def _shap_global_importance(fitted_pipeline, X_sample, max_samples=250):
    try:
        import shap
        transformed = fitted_pipeline.named_steps["preprocessor"].transform(
            fitted_pipeline.named_steps["outlier_handler"].transform(
                fitted_pipeline.named_steps["feature_engineer"].transform(X_sample.copy())
            )
        )
        selector = fitted_pipeline.named_steps.get("feature_selector")
        if selector is not None:
            transformed = selector.transform(transformed)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        if len(transformed) > max_samples:
            transformed = transformed[:max_samples]
        model = fitted_pipeline.named_steps["model"]
        explainer = shap.Explainer(model, transformed)
        values = explainer(transformed).values
        if values.ndim == 3:
            values = np.mean(np.abs(values), axis=2)
        importance = np.mean(np.abs(values), axis=0)
        names = _get_selected_feature_names(fitted_pipeline)
        if len(names) != len(importance):
            names = np.asarray([f"feature_{i}" for i in range(len(importance))])
        return pd.DataFrame({"Feature": names, "SHAP Importance": importance}).sort_values("SHAP Importance", ascending=False).head(20).reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=["Feature", "SHAP Importance"])

# ============================================================
# 11. MAIN AUTOML ENGINE
# ============================================================

def evaluate_autonomous_agent_pipeline(
    df,
    target_column,
    imbalance_method="No Treatment",
    training_mode="Fast",
    feature_engineering=False,
    outlier_method="No Treatment",
    feature_selection=False,
    feature_selection_percentile=80,
    shap_explainability=False,
    auto_metric_strategy=True,
    enable_ensemble=True,
):
    if df is None or df.empty:
        raise ValueError("The dataset is empty.")

    valid_imbalance_methods = {
        "No Treatment",
        "Class Weighting",
        "SMOTE",
        "Random Oversampling",
        "Random Undersampling",
    }
    if imbalance_method not in valid_imbalance_methods:
        raise ValueError(f"Unknown imbalance method: {imbalance_method}")

    if training_mode not in {"Fast", "Balanced", "Deep"}: training_mode = "Fast"
    if outlier_method not in {"No Treatment", "Winsorize"}: outlier_method = "No Treatment"
    feature_selection_percentile = int(max(10, min(100, feature_selection_percentile)))

    data = df.copy()
    logs = []
    tuning_explanations = {}
    best_model_artifact = None
    best_model_diagnostics = {}

    if target_column == "[Unsupervised Clustering]" or target_column is None:
        return _run_unsupervised(data, training_mode, logs)

    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' was not found.")

    data = data.dropna(subset=[target_column])
    if data.empty:
        raise ValueError("No rows remain after removing missing target values.")

    y = data[target_column].copy()
    X = _clean_features(data.drop(columns=[target_column]).copy())
    leakage_df = detect_target_leakage_risk(X, y, target_column)

    if y.nunique(dropna=True) < 2:
        raise ValueError("The target must contain at least two distinct values.")

    is_classification = detect_classification_target(y)
    config = get_training_configuration(training_mode)

    logs.append(f"⚙️ Training mode: {training_mode}")
    logs.append(f"📦 Rows used: {len(data):,}")
    logs.append(f"🧩 Base features used: {X.shape[1]:,}")
    logs.append(f"✨ Automatic Feature Engineering: {'ON' if feature_engineering else 'OFF'}")
    logs.append(f"📌 Outlier handling: {outlier_method}")
    logs.append(f"🧹 Automatic Feature Selection: {'ON' if feature_selection else 'OFF'}")
    logs.append(f"🧪 SHAP Explainability: {'ON' if shap_explainability else 'OFF'}")
    if not leakage_df.empty:
        logs.append(f"🚨 Leakage-risk detector flagged {len(leakage_df)} feature(s); no feature was removed automatically.")
    else:
        logs.append("✅ No strong target-leakage heuristic was detected.")

    # ========================================================
    # CLASSIFICATION
    # ========================================================
    if is_classification:
        task_type = "Autonomous Supervised Classification"
        balance_info = analyze_class_balance(y)
        primary_metric = "CV Macro F1"  # deterministic selection metric for all classification runs
        logs.append(f"🎯 Automatic metric strategy: {primary_metric}")
        logs.append(f"⚖️ Class balance status: {balance_info['status']}")
        if balance_info["ratio"] is not None:
            logs.append(f"📊 Majority/Minority ratio: {balance_info['ratio']:.2f}")

        y_codes, class_labels = pd.factorize(y, sort=True)
        X_train, X_test, y_train, y_test = _safe_classification_split(X, y_codes)
        cv_splits = _safe_cv_splits(y_train, config["cv"])
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)

        effective_imbalance_method = imbalance_method
        smote_k_neighbors = 1
        if imbalance_method == "SMOTE":
            smote_safe, smote_k_neighbors = _smote_settings(y_train, cv_splits)
            if not smote_safe:
                effective_imbalance_method = "Random Oversampling"
                logs.append("⚠️ Too few minority samples for SMOTE inside CV. Switched to Random Oversampling.")

        schema_features = AutomaticFeatureEngineer(enabled=feature_engineering).fit_transform(X_train)
        preprocessor = build_preprocessor(schema_features)
        model_configs = get_classification_models(effective_imbalance_method)
        selected_models = [m for m in config["classification_models"] if m in model_configs]

        if effective_imbalance_method == "Class Weighting":
            selected_models = [name for name in selected_models if model_configs[name].get("supports_class_weight")]

        results = []
        fitted_pipelines = {}

        for model_name in selected_models:
            spec = model_configs.get(model_name)
            try:
                base_model = clone(spec["model"])
                pipeline = build_imbalance_pipeline(
                    preprocessor=clone(preprocessor),
                    model=base_model,
                    imbalance_method=effective_imbalance_method,
                    smote_k_neighbors=smote_k_neighbors,
                    use_scaler=spec["scale"],
                    feature_engineering=feature_engineering,
                    outlier_method=outlier_method,
                    feature_selection=feature_selection,
                    feature_selection_percentile=feature_selection_percentile,
                    selection_score_func=mutual_info_classif,
                )
                current_n_iter = min(config["n_iter"], calculate_total_combinations(spec["params"]))
                search = RandomizedSearchCV(
                    estimator=pipeline, param_distributions=spec["params"], n_iter=current_n_iter,
                    scoring="f1_macro", cv=cv, n_jobs=-1, random_state=RANDOM_STATE, refit=True, error_score=np.nan,
                )
                search.fit(X_train, y_train)
                if np.isnan(search.best_score_): raise RuntimeError("Candidates failed.")
                predictions = search.predict(X_test)
                class_diag = _classification_diagnostics(y_test, predictions)

                results.append({
                    "Model Name": model_name,
                    "Treatment": effective_imbalance_method,
                    "CV Macro F1": round(search.best_score_ * 100, 2),
                    "Test Macro F1": round(f1_score(y_test, predictions, average="macro", zero_division=0) * 100, 2),
                    "Balanced Accuracy": round(balanced_accuracy_score(y_test, predictions) * 100, 2),
                    "Accuracy": round(accuracy_score(y_test, predictions) * 100, 2),
                    "Precision Macro": round(class_diag["precision_macro"], 2),
                    "Recall Macro": round(class_diag["recall_macro"], 2),
                    "Best Parameters": str(search.best_params_),
                })
                tuning_explanations[model_name] = explain_parameters(search.best_params_)
                fitted_pipelines[model_name] = search.best_estimator_
            except Exception as exc:
                logs.append(f"❌ {model_name} failed: {exc}")

        # Ensemble Voting
        if enable_ensemble and len(fitted_pipelines) >= 2:
            try:
                sorted_cands = sorted(results, key=lambda x: x[primary_metric], reverse=True)[:3]
                estimators = [(c["Model Name"], fitted_pipelines[c["Model Name"]]) for c in sorted_cands]
                ensemble = VotingClassifier(estimators=estimators, voting="soft" if all(hasattr(e[1], "predict_proba") for e in estimators) else "hard")
                ensemble.fit(X_train, y_train)
                ens_preds = ensemble.predict(X_test)
                ens_diag = _classification_diagnostics(y_test, ens_preds)
                results.append({
                    "Model Name": "🏆 Ensemble (Voting Top 3)",
                    "Treatment": effective_imbalance_method,
                    "CV Macro F1": round(sorted_cands[0]["CV Macro F1"], 2),
                    "Test Macro F1": round(f1_score(y_test, ens_preds, average="macro", zero_division=0) * 100, 2),
                    "Balanced Accuracy": round(balanced_accuracy_score(y_test, ens_preds) * 100, 2),
                    "Accuracy": round(accuracy_score(y_test, ens_preds) * 100, 2),
                    "Precision Macro": round(ens_diag["precision_macro"], 2),
                    "Recall Macro": round(ens_diag["recall_macro"], 2),
                    "Best Parameters": "Ensemble of best tuned candidates",
                })
                fitted_pipelines["🏆 Ensemble (Voting Top 3)"] = ensemble
            except Exception as ens_e:
                logs.append(f"⚠️ Ensemble skipped: {ens_e}")

        if not results: raise RuntimeError("All classification models failed.")
        # Model selection is based on cross-validation only. The held-out test set
        # is evaluation-only and can never decide the winner.
        cv_metric = "CV Macro F1"
        tie_breakers = ["CV Macro F1", "Model Name"]
        sort_cols = [cv_metric] + [c for c in tie_breakers if c != cv_metric and c in results[0]]
        ascending = [False] * (len(sort_cols) - 1) + [True] if sort_cols and sort_cols[-1] == "Model Name" else [False] * len(sort_cols)
        leaderboard_df = pd.DataFrame(results).sort_values(by=sort_cols, ascending=ascending).reset_index(drop=True)
        winner = leaderboard_df.iloc[0]["Model Name"]
        winner_score = float(leaderboard_df.iloc[0][cv_metric])

        pipe = fitted_pipelines.get(winner)
        pred = pipe.predict(X_test)
        best_model_artifact = {
            "pipeline": pipe, "task": "classification", "model_name": winner,
            "target": target_column, "class_labels": [str(x) for x in class_labels],
            "test_predictions": pred, "test_actual": np.asarray(y_test), "test_features": X_test.copy()
        }
        best_model_diagnostics = _classification_diagnostics(y_test, pred)
        best_model_diagnostics["feature_importance"] = _feature_importance_dataframe(pipe) if hasattr(pipe, "named_steps") else pd.DataFrame()
        best_model_diagnostics["leakage_risks"] = leakage_df
        if shap_explainability and hasattr(pipe, "named_steps"):
            shap_df = _shap_global_importance(pipe, X_test)
            best_model_diagnostics["shap_importance"] = shap_df

        logs.append(f"🏆 Best model by {primary_metric}: {winner}")
        logs.append(f"📈 {primary_metric}: {winner_score:.2f}%")
        log_experiment_to_db(task_type, target_column, winner, winner_score, training_mode, effective_imbalance_method, feature_engineering, outlier_method)

        return task_type, leaderboard_df, winner, f"{winner_score:.2f}%", logs, tuning_explanations, best_model_artifact, best_model_diagnostics

    # ========================================================
    # REGRESSION
    # ========================================================
    task_type = "Autonomous Supervised Regression"
    primary_metric = "Test R²"
    logs.append(f"🎯 Automatic metric strategy: {primary_metric}")

    y_numeric = pd.to_numeric(y, errors="coerce")
    valid_target = y_numeric.notna()
    X = X.loc[valid_target].copy()
    y_numeric = y_numeric.loc[valid_target].copy()

    if len(y_numeric) < 5: raise ValueError("Not enough valid numeric target rows for regression.")

    X_train, X_test, y_train, y_test = train_test_split(X, y_numeric, test_size=0.20, random_state=RANDOM_STATE)
    cv = KFold(n_splits=max(2, min(config["cv"], len(X_train))), shuffle=True, random_state=RANDOM_STATE)

    schema_features = AutomaticFeatureEngineer(enabled=feature_engineering).fit_transform(X_train)
    preprocessor = build_preprocessor(schema_features)
    model_configs = get_regression_models()
    selected_models = [m for m in config["regression_models"] if m in model_configs]

    results = []
    fitted_pipelines = {}

    for model_name in selected_models:
        spec = model_configs.get(model_name)
        try:
            steps = [
                ("feature_engineer", AutomaticFeatureEngineer(enabled=feature_engineering)),
                ("outlier_handler", QuantileOutlierHandler(method=outlier_method)),
                ("preprocessor", clone(preprocessor)),
            ]
            if feature_selection:
                steps.append(("feature_selector", SelectPercentile(score_func=mutual_info_regression, percentile=feature_selection_percentile)))
            if spec["scale"]:
                steps.append(("scaler", StandardScaler()))
            steps.append(("model", clone(spec["model"])))

            pipeline = SklearnPipeline(steps=steps)
            current_n_iter = min(config["n_iter"], calculate_total_combinations(spec["params"]))
            search = RandomizedSearchCV(
                estimator=pipeline, param_distributions=spec["params"], n_iter=current_n_iter,
                scoring="r2", cv=cv, n_jobs=-1, random_state=RANDOM_STATE, refit=True, error_score=np.nan,
            )
            search.fit(X_train, y_train)
            if np.isnan(search.best_score_): raise RuntimeError("Candidates failed.")
            predictions = search.predict(X_test)
            test_r2 = r2_score(y_test, predictions) * 100
            reg_diag = _regression_diagnostics(y_test, predictions)

            results.append({
                "Model Name": model_name,
                "CV R²": round(search.best_score_ * 100, 2),
                "Test R²": round(test_r2, 2),
                "MAE": round(reg_diag["MAE"], 4),
                "RMSE": round(reg_diag["RMSE"], 4),
                "Best Parameters": str(search.best_params_),
            })
            tuning_explanations[model_name] = explain_parameters(search.best_params_)
            fitted_pipelines[model_name] = search.best_estimator_
        except Exception as exc:
            logs.append(f"❌ {model_name} failed: {exc}")

    # Ensemble Voting Regressor
    if enable_ensemble and len(fitted_pipelines) >= 2:
        try:
            sorted_cands = sorted(results, key=lambda x: x["Test R²"], reverse=True)[:3]
            estimators = [(c["Model Name"], fitted_pipelines[c["Model Name"]]) for c in sorted_cands]
            ensemble = VotingRegressor(estimators=estimators)
            ensemble.fit(X_train, y_train)
            ens_preds = ensemble.predict(X_test)
            reg_diag = _regression_diagnostics(y_test, ens_preds)
            results.append({
                "Model Name": "🏆 Ensemble (Voting Top 3)",
                "CV R²": round(sorted_cands[0]["CV R²"], 2),
                "Test R²": round(r2_score(y_test, ens_preds) * 100, 2),
                "MAE": round(reg_diag["MAE"], 4),
                "RMSE": round(reg_diag["RMSE"], 4),
                "Best Parameters": "Ensemble of best tuned candidates",
            })
            fitted_pipelines["🏆 Ensemble (Voting Top 3)"] = ensemble
        except Exception as ens_e:
            logs.append(f"⚠️ Ensemble skipped: {ens_e}")

    if not results: raise RuntimeError("All regression models failed.")
    # Select by CV R²; Test R² remains a final holdout report only.
    leaderboard_df = pd.DataFrame(results).sort_values(by=["CV R²", "Test R²", "Model Name"], ascending=[False, False, True]).reset_index(drop=True)
    winner = leaderboard_df.iloc[0]["Model Name"]
    winner_score = float(leaderboard_df.iloc[0]["CV R²"])

    pipe = fitted_pipelines.get(winner)
    pred = pipe.predict(X_test)
    best_model_artifact = {
        "pipeline": pipe, "task": "regression", "model_name": winner, "target": target_column,
        "test_predictions": pred, "test_actual": np.asarray(y_test), "test_features": X_test.copy()
    }
    best_model_diagnostics = _regression_diagnostics(y_test, pred)
    best_model_diagnostics["feature_importance"] = _feature_importance_dataframe(pipe) if hasattr(pipe, "named_steps") else pd.DataFrame()
    best_model_diagnostics["leakage_risks"] = leakage_df
    if shap_explainability and hasattr(pipe, "named_steps"):
        shap_df = _shap_global_importance(pipe, X_test)
        best_model_diagnostics["shap_importance"] = shap_df

    logs.append(f"🏆 Best regression model: {winner}")
    logs.append(f"📈 Test R²: {winner_score:.2f}%")
    log_experiment_to_db(task_type, target_column, winner, winner_score, training_mode, imbalance_method, feature_engineering, outlier_method)

    return task_type, leaderboard_df, winner, f"{winner_score:.2f}%", logs, tuning_explanations, best_model_artifact, best_model_diagnostics

# ============================================================
# 12. PUBLIC PREDICTION / EXPORT
# ============================================================

def predict_with_artifact(model_artifact, X_new):
    if not model_artifact or "pipeline" not in model_artifact: raise ValueError("No fitted AutoML model artifact is available.")
    if X_new is None or len(X_new) == 0: raise ValueError("Prediction data is empty.")
    return model_artifact["pipeline"].predict(X_new.copy())

def save_model_artifact(model_artifact, filepath):
    import joblib
    if not model_artifact or "pipeline" not in model_artifact: raise ValueError("No fitted AutoML model artifact is available.")
    joblib.dump(model_artifact, filepath)
    return filepath

def load_model_artifact(filepath):
    import joblib
    artifact = joblib.load(filepath)
    if not isinstance(artifact, dict) or "pipeline" not in artifact: raise ValueError("Invalid AutoML model artifact.")
    return artifact