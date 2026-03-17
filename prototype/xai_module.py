import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import dice_ml
from dictionaries import variable_descriptions, variable_labels, variable_labels2
from alibi.explainers import AnchorTabular
from sklearn.tree import DecisionTreeClassifier, plot_tree
from anchor import anchor_tabular

def plot_global_shap(best_regressor, X_test_classification, X_test_regression, test_delayed, categorical_vars):
    predicted_severity = np.zeros(len(X_test_classification))
    predicted_severity[test_delayed.reset_index(drop=True).index] = best_regressor.predict(X_test_regression)

    explainer_cls = st.session_state.shap_explainer_classification
    shap_values_cls = explainer_cls(X_test_classification)

    explainer_reg = st.session_state.shap_explainer_regression
    shap_values_reg = explainer_reg(X_test_regression)

    shap_cls_values = shap_values_cls.values
    if shap_cls_values.ndim == 3:
        shap_cls_values = shap_cls_values[..., 1]

    shap_reg_values = shap_values_reg.values
    if shap_reg_values.ndim > 2:
        shap_reg_values = shap_reg_values[..., 0]

    expected_shap_values = np.zeros_like(shap_cls_values)
    delayed_mask = X_test_classification.index.isin(test_delayed.index)
    delayed_idx = np.where(delayed_mask)[0]

    expected_shap_values[delayed_idx] = (
        shap_cls_values[delayed_idx] * predicted_severity[delayed_idx, None]
        + shap_reg_values
    )

    shap_df = pd.DataFrame(expected_shap_values, columns=X_test_classification.columns)

    for cat_var in categorical_vars.keys():
        one_hot_cols = [col for col in X_test_classification.columns if col.startswith(cat_var + "_")]
        if one_hot_cols:
            shap_df[cat_var] = shap_df[one_hot_cols].sum(axis=1)
            shap_df.drop(columns=one_hot_cols, inplace=True)

    renamed_cols = [variable_labels.get(col, col) for col in shap_df.columns]
    shap_df.columns = renamed_cols

    plot_shap_df = shap_df.drop(columns=[col for col in shap_df.columns if col.startswith("project_type")], errors='ignore')
    mean_shap = plot_shap_df.abs().mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10,6))
    mean_shap.plot.bar(ax=ax, color="#ffab72ff")
    ax.set_title("Globale SHAP-waarden")
    ax.set_ylabel("Gemiddelde SHAP-waarde")
    plt.tight_layout()

    return fig, mean_shap.index.tolist()

def explain_global_shap(mean_shap_order, feature_descriptions):
    with st.expander("Wat betekent deze grafiek?"):
        st.markdown("""
        **Deze grafiek laat zien welke factoren de grootste invloed hebben op projectvertragingen binnen alle projecten in uw dataset.**

        - Elke balk vertegenwoordigd een factor die de voorspelde vertraging beïnvloedt.
        - Hogere balken = grotere impact op de voorspelling.
        - De waarden (SHAP-waarden) vertegenwoordigen het **gemiddelde effect** van elke factor.
        - De grafiek toont **hoe belangrijk** elke factor is, maar geeft **niet aan in welke richting** het effect werkt (dus niet of de vertraging toeneemt of afneemt). Dit is wel te zien in de **lokale SHAP-grafiek** op de **Voorspellen** pagina.
        - Dit is een **globale verklaring**, wat betekent dat het de werking van het model voor alle projecten in de dataset samenvat, en niet slechts één specifiek project.                  
        """)
    with st.expander("Uitleg variabelen"):
        for feature in mean_shap_order:
            description = feature_descriptions.get(feature)
            st.markdown(f"**{feature}**: {description}")

def generate_counterfactuals(X_train_regression, y_train_regression, best_regressor, input_data, features_to_vary, max_delay):
    query_instance = input_data.iloc[[0]]

    permitted_range = {
        "decision_making_structure": [1, 10],
        "contractor_experience": [1, 40],
        "num_subcontractors": [1, 40],
        "planning_detail": [50, 100],
        "design_maturity": [30, 100],
        "design_team_experience_years": [1, 40],
        "use_of_bim_1": [0, 1],
        "labour_shortage": [0, 40],
        "experienced_labour": [20, 100],
        "consultant_experience": [1, 30],
        "consultant_availability": [30, 100],
        "consultant_prior_collaboration_1": [0, 1],
        "material_supply_reliability": [1, 5],
        "equipment_availability": [0.3, 1.0],
        "average_equipment_age": [0, 25],
    }

    direction_change_allowed = {
        "decision_making_structure": "decreasing",
        "contractor_experience": "increasing",
        "num_subcontractors": "decreasing",
        "planning_detail": "increasing",
        "design_maturity": "increasing",
        "design_team_experience_years": "increasing",
        "use_of_bim_1": "increasing",
        "labour_shortage": "decreasing",
        "experienced_labour": "increasing",
        "consultant_experience": "increasing",
        "consultant_availability": "increasing",
        "consultant_prior_collaboration_1": "increasing",
        "material_supply_reliability": "increasing",
        "equipment_availability": "increasing",
        "average_equipment_age": "decreasing",
    }

    directional_range = {}
    for col in features_to_vary:
        val = query_instance.iloc[0][col]
        mn, mx = permitted_range[col]
        direction = direction_change_allowed[col]

        if direction == "increasing":
            directional_range[col] = [val, mx]
        elif direction == "decreasing":
            directional_range[col] = [mn, val]
        else:
            directional_range[col] = [mn, mx]

    binary_features = [
        "use_of_bim_1",
        "consultant_prior_collaboration_1"
    ]

    X_train_regression[binary_features] = (
        X_train_regression[binary_features]
        .astype(float)
        .round()
        .astype(int)
        .astype(str)
    )
    query_instance[binary_features] = (
        query_instance[binary_features]
        .astype(float)
        .round()
        .astype(int)
        .astype(str)
    )

    for col in binary_features:
        if col in directional_range:
            low, high = directional_range[col]

            low = int(float(low))
            high = int(float(high))

            directional_range[col] = [str(low), str(high)]
    
    categorical_features = ["use_of_bim_1", "consultant_prior_collaboration_1"]

    continuous_features = [
        col for col in X_train_regression.columns if col not in categorical_features
    ]

    Xy_train = X_train_regression.copy()
    Xy_train["delay_pct"] = y_train_regression

    data_dice = dice_ml.Data(
        dataframe=Xy_train,
        continuous_features=continuous_features,
        categorical_features=categorical_features,
        outcome_name="delay_pct"
    )
    model_dice = dice_ml.Model(
        model=best_regressor,
        backend="sklearn",
        model_type="regressor"
    )
    exp = dice_ml.Dice(data_dice, model_dice, method="random")

    dice_exp = exp.generate_counterfactuals(
        query_instance,
        total_CFs=5,
        desired_range=[0, max_delay],
        features_to_vary=features_to_vary,
        permitted_range=directional_range
    )

    cf_all = dice_exp.cf_examples_list[0].final_cfs_df.drop(columns=["delay_pct"], errors="ignore")

    cf_all["predicted_delay"] = cf_all.apply(lambda r: best_regressor.predict(pd.DataFrame([r]))[0], axis=1)


    best_cf = cf_all.loc[cf_all["predicted_delay"].idxmin()]

    differences = {}
    for col in features_to_vary:
        orig = query_instance[col].iloc[0]
        cf_val = best_cf[col]
        if col in categorical_features:
            differences[col] = f"{orig} → {cf_val}"
        else:
            differences[col] = cf_val - orig
    difference_series = pd.Series(differences)

    cf_display = pd.DataFrame({
        "Original": query_instance[features_to_vary].iloc[0],
        "Counterfactual": best_cf[features_to_vary],
        "Difference": difference_series
    })

    st.write(f"**Tegenfeitelijke variant met verlaagde voorspelde vertragingsernst ({best_cf['predicted_delay']:.3f})**")
    st.dataframe(cf_display, width=1200)

def generate_local_shap(shap_explainer, input_data, X_test_classification, categorical_vars):
    shap_values = shap_explainer(input_data)

    if shap_values.values.ndim == 3:
        shap_values_array = shap_values.values[:, :, 1]
    else:
        shap_values_array = shap_values.values

    shap_df = pd.DataFrame(shap_values_array, columns=X_test_classification.columns)

    for cat_var in categorical_vars.keys():
        one_hot_cols = [col for col in shap_df.columns if col.startswith(cat_var + "_")]
        if one_hot_cols:
            shap_df[cat_var] = shap_df[one_hot_cols].sum(axis=1)
            shap_df.drop(columns=one_hot_cols, inplace=True)

    renamed_cols = [variable_labels.get(col, col) for col in shap_df.columns]
    shap_df.columns = renamed_cols

    local_values = shap_df.iloc[0]
    fig, ax = plt.subplots(figsize=(10, 4))
    local_values.plot.bar(ax=ax, color="#ffab72ff")
    ax.set_title("Lokale SHAP-waarden")
    ax.set_ylabel("SHAP-waarde")
    st.pyplot(fig)
    plt.clf()

    with st.expander("Wat betekent deze grafiek?"):
        st.markdown("""
        **Deze grafiek toont welke factoren de grootste invloed hebben op projectvertragingen binnen alle projecten in uw dataset.**

        - Balken die **omhoog** gaan (positief) geven variabelen aan die de **voorspelde vertraging verhogen**.
        - Balken die **omlaag** gaan (negatief) geven variabelen aan die de **voorspelde vertraging verlagen**.
        - De lengte van de balk geeft de omvang van het effect weer.
        - Dit is een **lokale verklaring**, wat betekent dat deze alleen de voorspelling van dit project uitlegt en niet van de gehele dataset.
        """)

    with st.expander("Uitleg variabelen"):
            for key, label in variable_labels.items():
                description = variable_descriptions.get(key)
                st.markdown(f"**{label}** : {description}")

def generate_anchor_explanation1(model, X_train, input_instance, feature_names, categorical_features=None):

    def prettify_anchor_predicate(rule: str) -> str:
        """
        Make Anchor predicate readable:
        - Convert one-hot columns to: <Label> = <Category> or <Label> ≠ <Category>
        - Replace base feature names with Dutch labels from variable_labels2
        """
        import re

        # 1) Parse "<feature> <op> <value>"
        m = re.match(r"^(.+?)\s*([<>]=?)\s*([0-9]*\.?[0-9]+)$", rule.strip())
        if not m:
            return rule

        feature, op, val_str = m.group(1), m.group(2), m.group(3)

        # Try numeric conversion
        try:
            val = float(val_str)
        except ValueError:
            return rule

        # Helper to label a base feature
        def label_of(base: str) -> str:
            return variable_labels2.get(base, base)

        # 2) One-hot / dummy case: split on LAST underscore (fixes contract_award_method_*)
        # Example: "contract_award_method_Laagste prijs <= 0.00"
        # base="contract_award_method", category="Laagste prijs"
        if "_" in feature:
            base, category = feature.rsplit("_", 1)

            # Only treat as one-hot if it looks like a 0/1 dummy threshold
            if val in (0.0, 0.00, 0.5, 1.0):
                base_label = label_of(base)

                if op in ("<", "<=") and val <= 0.5:
                    return f"{base_label} ≠ {category}"
                if op in (">", ">=") and val >= 0.5:
                    return f"{base_label} = {category}"

        # 3) Default numeric feature case: "design_maturity > 69.62"
        feat_label = label_of(feature)

        # Keep your numeric formatting nice (optional)
        # If you want fewer decimals:
        if abs(val) < 1000:
            val_fmt = f"{val:.2f}"
        else:
            val_fmt = f"{val:.0f}"

        return f"{feat_label} {op} {val_fmt}"
    def predict_fn(x):
        return model.predict(x)

    explainer = AnchorTabular(
        predictor=predict_fn,
        feature_names=feature_names,
        categorical_names=categorical_features
    )

    explainer.fit(X_train.values)

    explanation = explainer.explain(
        input_instance.values,
        threshold=0.95
    )

    if explanation.anchor:
        st.write("**Beslisregel**")

        
        for r in explanation.anchor:
            st.markdown(f"- {prettify_anchor_predicate(r)}")


        st.markdown("---")

        st.markdown(
            f"**Betrouwbaarheid van deze regel:** {explanation.precision:.2f}  \n"
            "*In welk deel van de gevallen deze regel tot dezelfde voorspelling leidt.*"
        )

        st.markdown(
            f"**Toepasbaar op:** {explanation.coverage * 100:.1f}% van vergelijkbare projecten  \n"
            "*Hoe vaak deze combinatie van voorwaarden voorkomt bij soortgelijke projecten.*"
        )

        with st.expander("Wat betekent deze uitleg?"):
            st.markdown("""
                Deze uitleg toont een **regel** die laat zien onder welke omstandigheden het AI‑model
                zijn **inschatting van het risico op vertraging stabiel houdt**.

                **Hoe leest u deze regel?**
                - De voorwaarden hieronder vormen samen een **voldoende combinatie**.
                - **Zolang álle voorwaarden tegelijk gelden**, verandert de inschatting van het
                vertragingrisico **niet wezenlijk**.
                - Andere projectkenmerken mogen dan variëren, zonder dat dit leidt tot een andere
                beoordeling van het risico.

                👉 Zie het als:
                > *“Onder deze omstandigheden verandert de risicobeoordeling van het model doorgaans niet.”*
            """)
        with st.expander("Hoe kiest het model deze voorwaarden?"):
            st.markdown("""
                De AI heeft deze voorwaarden **niet vooraf gekregen**.

                Zo wordt de regel gevonden:
                1. Het model start bij dit specifieke project.
                2. Andere projectkenmerken worden vervolgens **willekeurig gevarieerd**.
                3. Het model controleert wanneer de inschatting van het vertragingsrisico **niet verandert**.
                4. Alleen voorwaarden die de voorspelling **stabiel houden** worden toegevoegd.
                5. De regel stopt zodra de voorspelling in **minstens 95% van de gevallen gelijk blijft**.

                De gebruikte drempelwaarden zijn dus punten waarop het model intern gevoelig wordt voor verandering""")

    else:
        st.warning(
            "Voor dit project kon geen stabiele beslisregel worden gevonden die met voldoende betrouwbaarheid geldt."
        )



def generate_anchor_explanation(
    model,
    X_train,
    input_instance,
    feature_names,
    categorical_features=None,   # (better name: categorical_names)
    class_names=None,
    threshold=0.95
):
    def prettify_anchor_predicate(rule: str) -> str:
        import re

        m = re.match(r"^(.+?)\s*([<>]=?)\s*([0-9]*\.?[0-9]+)$", rule.strip())
        if not m:
            return rule

        feature, op, val_str = m.group(1), m.group(2), m.group(3)

        try:
            val = float(val_str)
        except ValueError:
            return rule

        def label_of(base: str) -> str:
            return variable_labels2.get(base, base)

        # One-hot / dummy case
        if "_" in feature:
            base, category = feature.rsplit("_", 1)
            if val in (0.0, 0.5, 1.0):
                base_label = label_of(base)
                if op in ("<", "<=") and val <= 0.5:
                    return f"{base_label} ≠ {category}"
                if op in (">", ">=") and val >= 0.5:
                    return f"{base_label} = {category}"

        feat_label = label_of(feature)
        val_fmt = f"{val:.2f}" if abs(val) < 1000 else f"{val:.0f}"
        return f"{feat_label} {op} {val_fmt}"

    # --- predictor must return integer class labels (n,) ---
    def predict_fn(X: np.ndarray) -> np.ndarray:
        # model.predict should return labels for sklearn classifiers
        y = model.predict(X)
        # ensure integer array if possible
        try:
            return y.astype(int)
        except Exception:
            return y

    # Prepare training data as numpy
    X_train_np = X_train.values if hasattr(X_train, "values") else np.asarray(X_train)

    # Prepare instance as 1D numpy
    x_np = input_instance.values if hasattr(input_instance, "values") else np.asarray(input_instance)
    x_np = np.asarray(x_np)
    if x_np.ndim == 2:
        x_np = x_np[0]  # shape (n_features,)

    # Class names (optional but recommended for readability)
    if class_names is None:
        # If sklearn classifier with .classes_, use that; else fallback
        if hasattr(model, "classes_"):
            class_names = [str(c) for c in model.classes_]
        else:
            class_names = ["class_0", "class_1"]

    explainer = anchor_tabular.AnchorTabularExplainer(
        class_names,
        feature_names,
        X_train_np,
        categorical_names=(categorical_features or {})
    )

    # Explain single instance
    exp = explainer.explain_instance(x_np, predict_fn, threshold=threshold)

    # Extract rule strings + stats (anchor-exp exposes helper methods)
    # (Depending on version, these are methods; we call them.)
    rules = exp.names() if hasattr(exp, "names") else []
    precision = exp.precision() if hasattr(exp, "precision") else None
    coverage = exp.coverage() if hasattr(exp, "coverage") else None

    if rules:
        st.write("**Beslisregel**")
        for r in rules:
            st.markdown(f"- {prettify_anchor_predicate(r)}")

        st.markdown("---")

        if precision is not None:
            st.markdown(
                f"**Betrouwbaarheid van deze regel:** {precision:.2f}  \n"
                "*In welk deel van de gevallen deze regel tot dezelfde voorspelling leidt.*"
            )
        if coverage is not None:
            st.markdown(
                f"**Toepasbaar op:** {coverage * 100:.1f}% van vergelijkbare projecten  \n"
                "*Hoe vaak deze combinatie van voorwaarden voorkomt bij soortgelijke projecten.*"
            )

        with st.expander("Wat betekent deze uitleg?"):
            st.markdown("""
                Deze uitleg toont een **regel** die laat zien onder welke omstandigheden het AI‑model
                zijn **inschatting van het risico op vertraging stabiel houdt**.

                **Hoe leest u deze regel?**
                - De voorwaarden hieronder vormen samen een **voldoende combinatie**.
                - **Zolang álle voorwaarden tegelijk gelden**, verandert de inschatting van het
                vertragingrisico **niet wezenlijk**.
                - Andere projectkenmerken mogen dan variëren, zonder dat dit leidt tot een andere
                beoordeling van het risico.

                👉 Zie het als:
                > *“Onder deze omstandigheden verandert de risicobeoordeling van het model doorgaans niet.”*
            """)

        with st.expander("Hoe kiest het model deze voorwaarden?"):
            st.markdown(f"""
                De AI heeft deze voorwaarden **niet vooraf gekregen**.

                Zo wordt de regel gevonden:
                1. Het model start bij dit specifieke project.
                2. Andere projectkenmerken worden vervolgens **willekeurig gevarieerd**.
                3. Het model controleert wanneer de inschatting van het vertragingsrisico **niet verandert**.
                4. Alleen voorwaarden die de voorspelling **stabiel houden** worden toegevoegd.
                5. De regel stopt zodra de voorspelling in **minstens {int(threshold*100)}% van de gevallen gelijk blijft**.
            """)

    else:
        st.warning(
            "Voor dit project kon geen stabiele beslisregel worden gevonden die met voldoende betrouwbaarheid geldt."
        )

def generate_surrogate_tree(black_box_model, X_train, max_depth=3):

    y_surrogate = black_box_model.predict(X_train)

    surrogate = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=42
    )
    surrogate.fit(X_train, y_surrogate)

    fig, ax = plt.subplots(figsize=(20, 12))
    plot_tree(
        surrogate,
        feature_names=X_train.columns,
        class_names=["Geen vertraging", "Vertraging"],
        filled=True,
        rounded=True,
        fontsize=9,
        ax=ax
    )

    st.pyplot(fig,)

    with st.expander("Wat betekent deze uitleg?"):
        st.markdown("""
            Deze beslisboom is een **vereenvoudigde weergave** van hoe het AI‑model
            in grote lijnen beslissingen neemt.

            - De boom **benadert** het gedrag van het model, maar is niet het echte model.
            - Hij laat zien **welke factoren het model vaak als eerste gebruikt**.

            👉 Zie dit als:
            “Een globale indruk van hoe het model redeneert.”
""")
