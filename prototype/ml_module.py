import streamlit as st
import pandas as pd
import numpy as np
import time
import shap
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, root_mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, RandomForestRegressor, ExtraTreesRegressor, StackingClassifier, StackingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.naive_bayes import GaussianNB

def train_models(df, df_numerical, df_categorical, categorical_vars):
    start_time = time.time()
    
    st.write("### Modellen trainen...")

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=13, stratify=df["is_delayed"])

    numerical_cols = [c for c in df_numerical.columns if c != "planned_duration_days"]
    feature_df_cols = numerical_cols + list(df_categorical.columns)

    X_train_base = train_df[feature_df_cols].copy()
    X_test_base  = test_df[feature_df_cols].copy()

    y_train_classification = train_df["is_delayed"]
    y_test_classification  = test_df["is_delayed"]

    train_delayed = train_df[train_df["is_delayed"] == 1].copy()
    test_delayed  = test_df[test_df["is_delayed"] == 1].copy()

    y_train_regression = train_delayed["delay_pct"]
    y_test_regression  = test_delayed["delay_pct"]

    X_train_regression_base = train_delayed[feature_df_cols].copy()
    X_test_regression_base  = test_delayed[feature_df_cols].copy()

    categorical_cols = list(categorical_vars.keys())
    if "project_type" not in categorical_cols:
        categorical_cols.append("project_type")

    X_train_classification = pd.get_dummies(X_train_base, columns=categorical_cols, drop_first=True)
    X_test_classification  = pd.get_dummies(X_test_base,  columns=categorical_cols, drop_first=True)
    X_train_regression = pd.get_dummies(X_train_regression_base, columns=categorical_cols, drop_first=True)
    X_test_regression  = pd.get_dummies(X_test_regression_base,  columns=categorical_cols, drop_first=True)

    X_train_classification, X_test_classification = X_train_classification.align(X_test_classification, join='left', axis=1, fill_value=0)
    X_train_regression, X_test_regression = X_train_regression.align(X_test_regression, join='left', axis=1, fill_value=0)

    X_train_classification = X_train_classification.astype(float)
    X_test_classification  = X_test_classification.astype(float)
    X_train_regression = X_train_regression.astype(float)
    X_test_regression  = X_test_regression.astype(float)

    classification_models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, solver='liblinear'),
    "RandomForest": RandomForestClassifier(random_state=13),
    "ExtraTrees": ExtraTreesClassifier(random_state=13),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=13),
    "LightGBM": LGBMClassifier(random_state=13, verbose=-1),
    #"CatBoost": CatBoostClassifier(verbose=0, random_state=13),
    "MLP": MLPClassifier(max_iter=500, random_state=13),
    "GaussianNB": GaussianNB()
    }

    classification_param_grids = {
    "RandomForest": {"n_estimators": [100, 300], "max_depth": [None, 10, 20]},
    "ExtraTrees": {"n_estimators": [100, 300], "max_depth": [None, 10, 20]},
    "XGBoost": {"n_estimators": [100, 200], "max_depth": [3, 6], "learning_rate": [0.01, 0.1]},
    "LightGBM": {"n_estimators": [100, 200], "max_depth": [3, 6], "learning_rate": [0.01, 0.1]},
    #"CatBoost": {"iterations": [100, 200], "depth": [3, 6], "learning_rate": [0.01, 0.1]},
    "MLP": {"hidden_layer_sizes": [(50,), (100,)], "alpha": [0.0001, 0.001]}
    }

    stacked_classifier = StackingClassifier(
    estimators=[
    ("LogisticRegression", classification_models["LogisticRegression"]),
    ("RandomForest", classification_models["RandomForest"]),
    ("XGBoost", classification_models["XGBoost"]),
    ("MLP", classification_models["MLP"])
    ],
    final_estimator=LogisticRegression(max_iter=1000),
    n_jobs=-1
    )

    regression_models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(random_state=13),
    "ExtraTrees": ExtraTreesRegressor(random_state=13),
    "XGBoost": XGBRegressor(random_state=13),
    "LightGBM": LGBMRegressor(random_state=13, verbose=-1),
    #"CatBoost": CatBoostRegressor(verbose=0, random_state=13),
    "MLP": MLPRegressor(max_iter=500, random_state=13)
    }

    regression_param_grids = {
    "RandomForest": {"n_estimators": [100, 300], "max_depth": [None, 10, 20]},
    "ExtraTrees": {"n_estimators": [100, 300], "max_depth": [None, 10, 20]},
    "XGBoost": {"n_estimators": [100, 200], "max_depth": [3, 6], "learning_rate": [0.01, 0.1]},
    "LightGBM": {"n_estimators": [100, 200], "max_depth": [3, 6], "learning_rate": [0.01, 0.1]},
    #"CatBoost": {"iterations": [100, 200], "depth": [3, 6], "learning_rate": [0.01, 0.1]},
    "MLP": {"hidden_layer_sizes": [(50,), (100,)], "alpha": [0.0001, 0.001]}
    }

    stacked_regressor = StackingRegressor(
    estimators=[
    ("RandomForest", regression_models["RandomForest"]),
    ("XGBoost", regression_models["XGBoost"]),
    ("MLP", regression_models["MLP"])
    ],
    final_estimator=LinearRegression(),
    n_jobs=-1
    )

    def tune_or_fit(model_name, model, param_grid, X_tr, y_tr, is_classification=True):
        scoring = "roc_auc" if is_classification else "r2"
        if model_name in param_grid:
            grid = GridSearchCV(model, param_grid[model_name], cv=3, scoring=scoring, n_jobs=-1, verbose=0)
            grid.fit(X_tr, y_tr)
            return grid.best_estimator_, getattr(grid, "best_score_", None), grid.best_params_
        else:
            model.fit(X_tr, y_tr)
            return model, None, None
        
    classification_results = {}
    for name, model in classification_models.items():
        st.write(f"Classifier aan het trainen/tunen/testen: {name}")
        best_model, best_score, best_params = tune_or_fit(name, model, classification_param_grids, X_train_classification, y_train_classification, is_classification=True)
        y_proba = best_model.predict_proba(X_test_classification)[:,1]
        auc = roc_auc_score(y_test_classification, y_proba)
        classification_results[name] = {"model": best_model, "auc": auc, "best_params": best_params}
        st.write(f"  AUC: {auc:.3f}")

    st.write(f"Stacked classifier aan het trainen/testen")
    stacked_classifier.fit(X_train_classification, y_train_classification)
    y_proba_stack = stacked_classifier.predict_proba(X_test_classification)[:, 1]
    auc_stack = roc_auc_score(y_test_classification, y_proba_stack)
    classification_results['Stacked'] = {"model": stacked_classifier, "auc": auc_stack, "best_params": None}
    st.write(f"AUC: {auc_stack:.3f}")

    regression_results = {}
    for name, model in regression_models.items():
        st.write(f"Regressor aan het trainen/tunen/testen: {name}")
        if name in regression_param_grids:
            grid = GridSearchCV(model, regression_param_grids[name], cv=3, scoring="r2", n_jobs=-1, verbose=0)
            grid.fit(X_train_regression, y_train_regression)
            best_model = grid.best_estimator_
            best_params = grid.best_params_
        else:
            best_model = model.fit(X_train_regression, y_train_regression)
            best_params = None

        y_pred_regression = best_model.predict(X_test_regression)
        rmse = root_mean_squared_error(y_test_regression, y_pred_regression)
        regression_results[name] = {"model": best_model, "rmse": rmse, "best_params": best_params}
        st.write(f"RMSE: {rmse:.4f}")

    st.write(f"Stacked regressor aan het trainen/testen")
    stacked_regressor.fit(X_train_regression, y_train_regression)
    y_pred_stack = stacked_regressor.predict(X_test_regression)
    rmse_stack = root_mean_squared_error(y_test_regression, y_pred_stack)
    regression_results['Stacked'] = {"model": stacked_regressor, "rmse": rmse_stack, "best_params": None}
    st.write(f"RMSE: {rmse_stack:.4f}")

    best_classifier_name = max(classification_results.keys(), key=lambda k: classification_results[k]["auc"])
    best_classifier = classification_results[best_classifier_name]["model"]
    
    best_regressor_name = min(regression_results.keys(), key=lambda k: regression_results[k]["rmse"])
    best_regressor = regression_results[best_regressor_name]["model"]

    st.success(f"Beste classifier: {best_classifier_name} (AUC={classification_results[best_classifier_name]['auc']:.3f})")
    st.success(f"Beste regressor: {best_regressor_name} (RMSE={regression_results[best_regressor_name]['rmse']:.3f})")

    st.info(f"Training voltooid in {time.time() - start_time:.1f} seconden")

    if best_classifier_name in ["MLP", "Stacked"]:
        st.write(f"**De beste classifier is {best_classifier_name}, maar SHAP wordt niet ondersteund. Het op één na beste model wordt gebruikt voor SHAP.**")
        sorted_classifiers = sorted(classification_results.items(), key=lambda kv: kv[1]["auc"], reverse=True)
        second_best_name, second_best_info = sorted_classifiers[1]
        best_classifier_name = second_best_name
        best_classifier = second_best_info["model"]
        st.write(f"**{best_classifier_name} wordt gebruikt voor SHAP. (AUC={classification_results[best_classifier_name]['auc']:.3f})**")

    if best_regressor_name in ["MLP", "Stacked"]:
        st.write(f"**De beste regressor is {best_regressor_name}, maar SHAP wordt niet ondersteund. Het op één na beste model wordt gebruikt voor SHAP.**")
        sorted_regressors = sorted(regression_results.items(), key=lambda kv: kv[1]["rmse"])
        second_best_name, second_best_info = sorted_regressors[1]
        best_regressor_name = second_best_name
        best_regressor = second_best_info["model"]
        st.write(f"**{best_regressor_name} wordt gebruikt voor SHAP (RMSE={regression_results[best_regressor_name]['rmse']:.4f})**")

    if best_classifier_name in ["RandomForest", "XGBoost", "LightGBM", "CatBoost", "ExtraTrees"]:
        st.session_state.shap_explainer_classification = shap.TreeExplainer(best_classifier)
    else:
        st.session_state.shap_explainer_classification = shap.Explainer(best_classifier.predict_proba, X_train_classification)

    if best_regressor_name in ["RandomForest", "XGBoost", "LightGBM", "CatBoost", "ExtraTrees"]:
        st.session_state.shap_explainer_regression = shap.TreeExplainer(best_regressor)
    elif best_regressor_name == "LinearRegression":
        st.session_state.shap_explainer_regression = shap.LinearExplainer(best_regressor, X_train_regression)
    else:
        st.session_state.shap_explainer_regression = shap.Explainer(best_regressor)


    return best_classifier, best_regressor, X_train_classification, y_train_classification, X_train_regression, y_train_regression, X_test_classification, X_test_regression, test_delayed, categorical_vars, numerical_cols, categorical_cols

def generate_prediction(input_data, best_classifier, best_regressor, categorical_vars,
                        X_test_classification_columns, X_test_regression_columns, planned_duration_days):

    df_num = input_data.select_dtypes(include=[np.number])
    df_cat = input_data.select_dtypes(exclude=[np.number])

    df_cat_encoded = pd.DataFrame()
    for col, (choices, probs) in categorical_vars.items():
        for opt in choices:
            df_cat_encoded[f"{col}_{opt}"] = (input_data[col] == opt).astype(int)

    df_final = pd.concat([df_num, df_cat_encoded], axis=1)

    for col in X_test_classification_columns:
        if col not in df_final.columns:
            df_final[col] = 0
    df_final = df_final[X_test_classification_columns]
    X_input_classification = df_final.copy()

    for col in X_test_regression_columns:
        if col not in df_final.columns:
            df_final[col] = 0
    df_final = df_final[X_test_regression_columns]
    X_input_regression = df_final.copy()

    delay_class = best_classifier.predict(X_input_classification)[0]
    prob_delay = best_classifier.predict_proba(X_input_classification)[:, 1]
    predicted_severity = best_regressor.predict(X_input_regression)
    expected_delay_pct = prob_delay * predicted_severity
    expected_delay_days = expected_delay_pct * planned_duration_days

    return {
        "delay_class": int(delay_class),
        "probability_delay": prob_delay[0],
        "predicted_severity": predicted_severity[0],
        "expected_delay_pct": expected_delay_pct[0],
        "expected_delay_days": expected_delay_days[0]
    }