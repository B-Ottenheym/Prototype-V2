import streamlit as st
import pandas as pd
import numpy as np
from data_module import generate_synthetic_data, generate_template_excel
from ml_module import train_models, generate_prediction
from xai_module import plot_global_shap, explain_global_shap, generate_local_shap, generate_counterfactuals, generate_anchor_explanation, generate_surrogate_tree
from dictionaries import variable_descriptions, variable_groups, variable_labels, variable_labels2
from experiment_config import EXPERIMENT_MODE
from experiment_flow import run_experiment

st.set_page_config(page_title=None, page_icon=None, layout="centered", menu_items=None)

if EXPERIMENT_MODE:
    run_experiment()
    st.stop()

if "page" not in st.session_state:
    st.session_state.page = "home"

def go_to(page_name):
    st.session_state.page = page_name

if st.session_state.page == "home":
    #st.title("Explainable AI (XAI) prototype voor het voorspellen van vertragingen in bouwprojecten")
    st.markdown("<h1 style='text-align: center; color: black;'>Explainable AI (XAI) prototype voor het voorspellen van vertragingen in bouwprojecten</h1>", unsafe_allow_html=True)
    st.write("\n\n")

    col1, col2, col3 = st.columns([2, 2, 1])
    with col2:
        st.write("")
        if st.button("Model Trainen", key="train_button"):
            go_to("train")
        st.write("")
        if st.button("Voorspellen", key="predict_button"):
            go_to("predict")

elif st.session_state.page == "train":
    st.title("Model Trainen")
    
    colA, colB, space = st.columns([1, 1, 1])
    with colA:
        if st.button("⬅ Terug naar Home"):
            go_to("home")
    with colB:
        if st.button("➡ Ga naar Voorspellen"):
            go_to("predict")
    
    train_option = st.radio(
        "Kies trainingstype:",
        ["Genereer synthetische data", "Train model met eigen data"]
    )

    if train_option == "Genereer synthetische data":
        n_samples = st.number_input(
                        label = "Aantal projecten:",
                        value=2000,
                        step=1,
                        help="Het aantal gewenste projecten in de synthetische dataset."
                        )
        if st.button("Genereer en train"):
            with st.spinner("Data genereren en modellen trainen..."):
                df, df_numerical, df_categorical, categorical_vars = generate_synthetic_data(n_samples)
                st.session_state.df_numerical = df_numerical
                st.session_state.df_categorical = df_categorical
                best_classifier, best_regressor, X_train_classification, y_train_classification, X_train_regression, y_train_regression, X_test_classification, X_test_regression, test_delayed, categorical_vars, numerical_cols, categorical_cols = train_models(df, df_numerical, df_categorical, categorical_vars)
            st.session_state.best_classifier = best_classifier
            st.session_state.best_regressor = best_regressor
            st.session_state.categorical_vars = categorical_vars
            st.session_state.X_test_regression = X_test_regression
            st.session_state.X_test_classification = X_test_classification
            st.session_state.X_train_classification = X_train_classification
            st.session_state.y_train_classification = y_train_classification
            st.session_state.X_train_regression = X_train_regression
            st.session_state.y_train_regression = y_train_regression
            st.session_state.numerical_cols = numerical_cols
            st.session_state.categorical_cols = categorical_cols
            st.success("De modellen zijn succesvol getraind!")

            fig, shap_order = plot_global_shap(best_regressor, X_test_classification, X_test_regression, test_delayed, categorical_vars)
            st.pyplot(fig)

            explain_global_shap(shap_order, variable_descriptions)
            
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("➡ Ga naar Voorspellen", key="trainen_nr_voorspellen"):
                    go_to("predict")
            

    elif train_option == "Train model met eigen data":
        st.info("U kunt de template downloaden om te garanderen dat uw bestand de juiste structuur heeft.")
        if st.button("Download Data Template"):
            file_name = generate_template_excel()  
            with open(file_name, "rb") as f:
                excel_bytes = f.read()

            st.download_button(
                label="Download Excel Template",
                data=excel_bytes,
                file_name='project_data_template.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )

        uploaded_file = st.file_uploader("Upload Excel bestand", type=["xlsx"])
        if uploaded_file is not None:
            df_own = pd.read_excel(uploaded_file)
            st.write("Uw data:")
            st.dataframe(df_own.head())

            numerical_cols = df_own.select_dtypes(include=["int64", "float64"]).columns.tolist()
            categorical_cols = df_own.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

            categorical_vars = {col: df_own[col].dropna().unique().tolist() for col in categorical_cols}

            df_numerical = df_own[numerical_cols]
            df_categorical = df_own[categorical_cols]
            st.session_state.df_numerical = df_numerical
            st.session_state.df_categorical = df_categorical

            best_classifier, best_regressor, X_train_classification, y_train_classification, X_train_regression, y_train_regression, X_test_classification, X_test_regression, test_delayed, categorical_vars, numerical_cols, categorical_cols = train_models(df_own, df_numerical, df_categorical, categorical_vars)
            st.session_state.best_classifier = best_classifier
            st.session_state.best_regressor = best_regressor
            st.session_state.categorical_vars = categorical_vars
            st.session_state.X_test_classification = X_test_classification
            st.session_state.X_test_regression = X_test_regression
            st.session_state.X_train_classification = X_train_classification
            st.session_state.y_train_classification = y_train_classification

            fig, shap_order = plot_global_shap(best_regressor, X_test_classification, X_test_regression, test_delayed, categorical_vars)
            st.pyplot(fig)

            explain_global_shap(shap_order, variable_descriptions)

            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("➡ Ga naar Voorspellen", key="trainen_nr_voorspellen2"):
                    go_to("predict")

elif st.session_state.page == "predict":
    st.title("Voorspellen")
    if st.button("⬅ Terug naar Home"):
        go_to("home")

    st.write("Voer hieronder de projectdetails in:")

    if "best_classifier" not in st.session_state or "best_regressor" not in st.session_state:
        st.warning("Train eerst de modellen.")
        if st.button("➡ Ga naar Model Trainen"):
                go_to("train")
    else:
        user_inputs_num = {}
        user_inputs_cat = {}

        for section, variables in variable_groups.items():
            st.subheader(section)

            for var in variables:
                label = variable_labels.get(var, var)
                tooltip = variable_descriptions.get(var)

                integer_vars = ["num_subcontractors", "planned_duration_days", "contract_value_million"]

                if var in st.session_state.df_numerical.columns:
                    default_value = round(st.session_state.df_numerical[var].mean())
                    default_value = float(default_value)

                    user_inputs_num[var] = st.number_input(
                        label,
                        value=default_value,
                        step=0.01,
                        help=tooltip
                    )
                elif var in st.session_state.df_categorical.columns:
                    options = st.session_state.categorical_vars[var][0]
                    user_inputs_cat[var] = st.selectbox(label, options, help=tooltip)

        input_data_dict = {**user_inputs_num, **user_inputs_cat}
        input_data = pd.DataFrame({k: [v] for k, v in input_data_dict.items()})
        input_num = input_data[st.session_state.numerical_cols]
        input_cat = input_data[st.session_state.categorical_cols]
        input_cat_encoded = pd.get_dummies(input_cat, columns=st.session_state.categorical_cols, drop_first=True)
        input_encoded = pd.concat([input_num, input_cat_encoded], axis=1)
        input_encoded = input_encoded.reindex(columns=st.session_state.X_train_regression.columns, fill_value=0)
        input_encoded = input_encoded.astype(float)

        if st.button("Voorspellen"):
            result = generate_prediction(
                input_data=input_data,
                best_classifier=st.session_state.best_classifier,
                best_regressor=st.session_state.best_regressor,
                categorical_vars=st.session_state.categorical_vars,
                X_test_classification_columns=st.session_state.X_test_classification.columns,
                X_test_regression_columns=st.session_state.X_test_regression.columns,
                planned_duration_days=input_data["planned_duration_days"]
            )
            st.session_state.predicted_severity = result['predicted_severity']

            st.write("### Voorspellingsresultaten")
            st.write(f"Waarschijnlijkheid van vertraging: {result['probability_delay'] * 100:.1f}%")
            st.write(f"Ernst van vertraging: {result['predicted_severity'] * 100:.1f}%")
            st.write(f"Verwachte vertraging (waarschijnlijkheid x ernst): {result['expected_delay_pct'] * 100:.1f}%")
            st.write(f"Verwachte vertraging: {int(round(result['expected_delay_days']))} dagen")

            st.write("### Lokale SHAP verklaring")

            generate_local_shap(
                shap_explainer=st.session_state.shap_explainer_classification, 
                input_data=input_encoded, 
                X_test_classification=st.session_state.X_test_classification, 
                categorical_vars=st.session_state.categorical_vars)

            @st.fragment
            def cf():
                st.write("### Tegenfeitelijke verklaring (Actiegerichte wat-als-uitleg)")

                with st.expander("Wat zijn tegenfeitelijke verklaringen?"):
                    st.markdown("""
                    **Tegenfeitelijke verklaringen laten zien hoe u waarden van projectkenmerken kunt wijzigen om de voorspelling van het model te beinvloeden.**
                    - Elke tegenfeitelijke variant wordt gegenereerd door uitsluitend drie geselecteerde kenmerken te wijzigen en geeft mogelijke aanpassingen weer die de voorspelde vertraging kunnen verminderen.
                    - De tabel toont de oorspronkelijke waarde, de tegenfeitelijke waarde en het verschil.
                    - Boven de tabel wordt de nieuw voorspelde ernst van vertraging weergegeven.
                    """)

                with st.expander("Uitleg variabelen"):
                    for key, label in variable_labels.items():
                        description = variable_descriptions.get(key)
                        st.markdown(f"**{label}** : {description}")

                if 'selected_features' not in st.session_state:
                    st.session_state.selected_features = []

                allowed_variables = [
                    "decision_making_structure", "contractor_experience", "num_subcontractors",
                    "planning_detail", "design_maturity", "design_team_experience_years",
                    "use_of_bim_1", "labour_shortage", "experienced_labour", "consultant_experience",
                    "consultant_availability", "consultant_prior_collaboration_1",
                    "material_supply_reliability", "equipment_availability", "average_equipment_age"
                ]

                allowed_labels = [variable_labels2[v] for v in allowed_variables]

                default_labels = [variable_labels2[v] for v in st.session_state.selected_features if v in variable_labels2]

                st.markdown("""<style>div[data-baseweb="select"] * {color: black !important;}</style>""", unsafe_allow_html=True)
                selected_labels = st.multiselect(
                    "Selecteer exact drie kenmerken om te wijzigen:",
                    options=allowed_labels,
                    default=default_labels
                )

                max_delay = st.number_input(
                    "Stel de maximale vertragingsernst in (schaal 0-1):",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.predicted_severity,
                    step=0.01,
                    help="Als er geen tegenfeitelijke variant wordt gegenereerd, probeer dan deze waarde te verhogen of andere variabelen te selecteren. De beginwaarde in het onderstaande veld is de oorspronkelijk voorspelde vertragingsernst."
                )

                label_to_var = {v: k for k, v in variable_labels2.items()}
                selected_features = [label_to_var[label] for label in selected_labels]

                st.session_state.selected_features = selected_features

                if st.button("Genereer tegenfeitelijke verklaringen"):
                    if len(st.session_state.selected_features) != 3:
                        st.warning("Selecteer alstublieft exact drie kenmerken.")
                    else:
                        st.write(f"Bezig met genereren van tegenfeitelijke varianten voor: {st.session_state.selected_features}")
                        generate_counterfactuals(
                            X_train_regression=st.session_state.X_train_regression,
                            y_train_regression=st.session_state.y_train_regression,
                            best_regressor=st.session_state.best_regressor,
                            input_data=input_encoded,
                            features_to_vary=st.session_state.selected_features,
                            max_delay=max_delay
                        )
            cf()
            
            st.write("### Regel-gebaseerde verklaring (Anchors)")

            generate_anchor_explanation(
                model=st.session_state.best_classifier,
                X_train=st.session_state.X_train_classification,
                input_instance=input_encoded,
                feature_names=st.session_state.X_train_classification.columns.tolist()
            )

            st.write("### Globale beslisstructuur (surrogaatmodel)")
            generate_surrogate_tree(
                black_box_model=st.session_state.best_classifier,
                X_train=st.session_state.X_train_classification,
                max_depth=4
            )
