from __future__ import annotations

import uuid
import urllib.parse
from dataclasses import asdict

import numpy as np
import pandas as pd
import streamlit as st

from dictionaries import variable_groups, variable_labels, variable_descriptions
from experiment_config import (
    CONDITIONS,
    QUALTRICS_BASE_URL,
    SCENARIOS,
    artifact_path,
)

def _init_participant_state():
    if "pid" not in st.session_state: #participant id
        st.session_state.pid = str(uuid.uuid4())
    if "scenario_id" not in st.session_state:
        st.session_state.scenario_id = SCENARIOS[0].scenario_id
    if "condition" not in st.session_state:
        idx = abs(hash(st.session_state.pid)) % len(CONDITIONS)
        st.session_state.condition = CONDITIONS[idx]
    if "exp_step" not in st.session_state:
        st.session_state.exp_step = 1


def _get_scenario():
    sid = st.session_state.scenario_id
    for s in SCENARIOS:
        if s.scenario_id == sid:
            return s
    return SCENARIOS[0]


def _next():
    st.session_state.exp_step += 1


def _back():
    st.session_state.exp_step = max(1, st.session_state.exp_step - 1)


def _progress():
    st.progress((st.session_state.exp_step - 1) / 4)
    st.caption(f"Stap {st.session_state.exp_step} of 4")


def _build_qualtrics_url():
    params = {
        "pid": st.session_state.pid,
        "cond": st.session_state.condition,
    }
    qs = urllib.parse.urlencode(params)
    if "?" in QUALTRICS_BASE_URL:
        return QUALTRICS_BASE_URL + "&" + qs
    return QUALTRICS_BASE_URL + "?" + qs


def _features_to_table(features: dict) -> pd.DataFrame:
    rows = []
    for group, vars_ in variable_groups.items():
        for v in vars_:
            if v in features:
                rows.append({
                    "Categorie": group,
                    "Variabele": variable_labels.get(v, v),
                    "Waarde": features[v],
                    "Beschrijving": variable_descriptions.get(v, ""),
                })
    for k, val in features.items():
        if not any(k in vs for vs in variable_groups.values()):
            rows.append({
                "Categorie": "Overig",
                "Variabele": variable_labels.get(k, k),
                "Waarde": val,
                "Beschrijving": variable_descriptions.get(k, ""),
            })
    return pd.DataFrame(rows)

def step_1_consent():
    st.header("Welkom")
    st.markdown(
        """
In dit onderzoek maakt u kennis met een prototype van een AI‑gebaseerd
beslissingsondersteunend systeem voor bouwprojecten.

U krijgt een projectsituatie te zien, samen met een voorspelling van het risico op
projectvertraging die door het systeem wordt gegenereerd. Afhankelijk van de versie
van het systeem die u te zien krijgt, wordt deze voorspelling mogelijk ondersteund
door aanvullende uitleg.

Tijdens het experiment wordt u gevraagd om de informatie die het systeem presenteert
zorgvuldig te bekijken. Het onderzoek richt zich niet op het nemen van beslissingen,
maar op **uw perceptie van de uitkomsten van het systeem en de bijbehorende uitleg**.

Het experiment bestaat uit enkele korte stappen en neemt slechts enkele minuten in
beslag. Na afloop wordt u automatisch doorgestuurd naar een vragenlijst waarin u wordt
gevraagd uw ervaringen met het systeem te beoordelen.

Uw deelname is vrijwillig en uw antwoorden worden anoniem verwerkt. U kunt op elk
moment stoppen met het experiment zonder opgave van reden.
"""
    )

    consent = st.checkbox(
        "Ik heb de bovenstaande informatie gelezen en ga akkoord met deelname aan dit onderzoek."
    )

    col1, col2 = st.columns([1, 1])
    with col2:
        st.button("Volgende", key="step1_next", disabled=not consent, on_click=_next)


def step_2_assignment():
    st.header("Uitleg van het systeem")

    st.markdown(
        """
In de volgende stap krijgt u een projectsituatie te zien, samen met een voorspelling
van het risico op projectvertraging die door een AI‑systeem wordt gegenereerd.

U bent toegewezen aan een specifieke versie van het systeem. Deze versie verschilt in
de manier waarop de voorspelling wordt toegelicht. Hieronder wordt kort uitgelegd hoe
de uitleg in uw versie is opgebouwd.
"""
    )

    cond = st.session_state.condition
    st.info(f"Toegewezen versie: **{st.session_state.condition}**")
    if cond == "Black box":
        st.info(
            """
**In deze versie van het systeem wordt alleen de voorspelling getoond.**

Er wordt geen aanvullende uitleg gegeven over hoe het systeem tot deze voorspelling
is gekomen.
"""
        )

    elif cond == "SHAP":
        st.info(
            """
**In deze versie van het systeem wordt de voorspelling ondersteund door een visuele uitleg.**

De uitleg laat zien welke projectkenmerken volgens het systeem het meest hebben
bijgedragen aan de voorspelling, en in welke mate deze kenmerken het risico op
vertraging verhogen of verlagen.
"""
        )

    elif cond == "Anchors":
        st.info(
            """
**In deze versie van het systeem wordt de voorspelling toegelicht met behulp van regels.**

Deze regels beschrijven combinaties van projectkenmerken waarvoor de voorspelling
geldig is. De uitleg geeft inzicht in welke voorwaarden doorslaggevend zijn geweest
voor de uitkomst.
"""
        )

    elif cond == "Tegenfeitelijk":
        st.info(
            """
**In deze versie van het systeem wordt de voorspelling toegelicht met alternatieve scenario’s.**

De uitleg laat zien hoe kleine aanpassingen in specifieke projectkenmerken zouden
kunnen leiden tot een andere voorspelling, bijvoorbeeld een lager risico op
vertraging.
"""
        )

    elif cond == "Surrogaatmodel (beslisboom)":
        st.info(
            """
**In deze versie van het systeem wordt de voorspelling toegelicht met een vereenvoudigd model.**

Dit model geeft een overzicht van de belangrijkste beslisregels die het AI‑systeem
gebruikt om tot een voorspelling te komen, in een vorm die makkelijker te interpreteren is.
"""
        )

    st.markdown(
        """
Lees deze uitleg zorgvuldig door. In de volgende stap ziet u de projectsituatie en
kunt u de voorspelling en bijbehorende uitleg bekijken.
"""
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        st.button("Terug", key="step2_back", on_click=_back)
    with col2:
        st.button("Volgende", key="step2_next", on_click=_next)


def step_3_scenario():
    scenario = _get_scenario()
    st.header("Scenario en voorspelling")
    st.subheader(scenario.title)
    st.markdown(scenario.narrative_markdown)

    if scenario.image_path:
        try:
            st.image(scenario.image_path, use_container_width=True)
        except Exception:
            st.warning("De scenario-afbeelding kon niet worden geladen.")

    st.markdown("#### Projectkenmerken")
    df = _features_to_table(scenario.features)

    
    df = df.copy()
    df = df.reset_index(drop=True)
    
    for col in df.columns:
        df[col] = df[col].astype(str)
    
    st.dataframe(df)


    st.markdown("---")
    st.markdown(
        """Hieronder kunt u op **Voorspellen** klikken om de AI‑uitkomst en de bijbehorende uitleg te bekijken.

De invoerwaarden zijn vastgezet voor dit onderzoek en kunnen niet worden aangepast."""
    )

    if "show_results" not in st.session_state:
        st.session_state.show_results = False

    if st.button("Voorspellen", key="step3_predict",):
        st.session_state.show_results = True

    if st.session_state.show_results:
        prob_delay = 0.62
        severity = 0.18
        expected_delay_pct = prob_delay * severity
        expected_delay_days = expected_delay_pct * float(
            scenario.features.get("planned_duration_days", 365)
        )

        st.subheader("AI‑uitkomst")
        st.write(f"Waarschijnlijkheid van vertraging: **{prob_delay * 100:.1f}%**")
        st.write(f"Ernst van vertraging: **{severity * 100:.1f}%**")
        st.write(f"Verwachte vertraging (waarschijnlijk × ernst): **{expected_delay_pct * 100:.1f}%**")
        st.write(f"Verwachte vertraging: **{expected_delay_days:.0f} dagen**")

        st.subheader("Uitleg")
        cond = st.session_state.condition

        if cond == "Black box":
            st.info("In deze versie van het systeem wordt geen uitleg bij de voorspelling gegeven.")

        elif cond in ("SHAP", "Surrogaatmodel (beslisboom)"):
            p = artifact_path(scenario.scenario_id, cond)
            if p.exists():
                st.image(str(p), use_container_width=True)
            else:
                st.warning(f"Afbeelding niet gevonden: {p}")

        elif cond == "Anchors":
            p = artifact_path(scenario.scenario_id, cond)
            if p.exists():
                st.markdown(p.read_text(encoding="utf-8"))
            else:
                st.warning(f"Anchor regels niet gevonden: {p}")

        elif cond == "Tegenfeitelijk":
            p = artifact_path(scenario.scenario_id, cond)
            if p.exists():
                st.dataframe(pd.read_csv(p), use_container_width=True)
            else:
                st.warning(f"Tabel niet gevonden: {p}")

        if cond == "SHAP":
            with st.expander("Wat betekent deze grafiek?"):
                st.markdown("""
                Deze grafiek laat zien **welke projectkenmerken volgens het model de grootste invloed hebben**
                op de voorspelling voor **dit specifieke project**.

                - Balken die **omhoog** wijzen geven kenmerken aan die het risico op vertraging **verhogen**.
                - Balken die **omlaag** wijzen geven kenmerken aan die het risico op vertraging **verlagen**.
                - De lengte van de balk geeft aan **hoe groot die invloed is**.

                Dit is een **lokale uitleg**:  
                deze verklaart alleen de voorspelling van dit project, niet van alle projecten in de dataset.
                """)

        elif cond == "Anchors":
            with st.expander("Wat betekent deze uitleg?"):
                st.markdown("""
                Deze uitleg toont een **regel** die beschrijft onder welke omstandigheden
                het AI‑model zijn inschatting van het risico op projectvertraging **stabiel houdt**.

                **Hoe leest u deze regel?**
                - De genoemde voorwaarden vormen samen een **voldoende combinatie**.
                - **Zolang alle voorwaarden tegelijk gelden**, verandert de inschatting van het
                vertragingrisico **niet wezenlijk**.
                - Andere projectkenmerken mogen dan variëren zonder dat dit leidt tot een andere
                inschatting.

                👉 Zie dit als:  
                *“Onder deze omstandigheden blijft de risicobeoordeling van het model doorgaans gelijk.”*
                """)

            with st.expander("Hoe komt deze regel tot stand?"):
                st.markdown("""
                    De AI heeft deze voorwaarden **niet vooraf gekregen**.

                    De regel wordt als volgt bepaald:
                    1. Het model start bij deze specifieke projectsituatie.
                    2. Andere projectkenmerken worden vervolgens **gevarieerd**.
                    3. Het model controleert wanneer de inschatting van het risico **gelijk blijft**.
                    4. Alleen voorwaarden die de voorspelling **stabiel houden** worden opgenomen.
                    5. De regel stopt zodra de voorspelling in **minstens 95% van de gevallen hetzelfde blijft**.

                    De gebruikte grenswaarden geven dus punten aan waarop het model gevoelig wordt voor verandering.
                    """)

        elif cond == "Tegenfeitelijk":
            with st.expander("Wat betekent deze uitleg?"):
                st.markdown("""
                    Tegenfeitelijke verklaringen laten zien **hoe kleine aanpassingen in projectkenmerken**
                    zouden kunnen leiden tot een andere voorspelling van het model.

                    - Elke tegenfeitelijke variant wijzigt **een beperkt aantal kenmerken**.
                    - De tabel toont de oorspronkelijke waarde, de aangepaste waarde en het verschil.
                    - Deze varianten geven mogelijke richtingen weer waarin het risico op vertraging lager zou uitvallen.

                    De uitleg laat dus zien **welke veranderingen volgens het model relevant zijn** voor de voorspelling.
                    """)

        elif cond == "Surrogaatmodel (beslisboom)":
            with st.expander("Wat betekent deze uitleg?"):
                st.markdown("""
                    Deze beslisboom is een **vereenvoudigde weergave** van hoe het AI‑model
                    in grote lijnen tot een voorspelling komt.

                    - De boom **benadert** het gedrag van het oorspronkelijke model, maar is niet hetzelfde.
                    - Hij laat zien **welke projectkenmerken vaak als eerste worden gebruikt** bij het maken van een inschatting.

                    👉 Zie dit als:  
                    *Een globale indruk van hoe het model redeneert.*
                    """)

    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.button("Terug", key="step3_back", on_click=_back)
    with col2:
        st.button(
            "Doorgaan naar vragenlijst", key="step3_next",
            disabled=not st.session_state.get("show_results", False),
            on_click=_next
        )

def step_4_redirect():
    st.header("Vragenlijst")
    st.markdown(
        """Klik op de knop hieronder om door te gaan naar de vragenlijst.\n\n"""
    )
    url = _build_qualtrics_url()
    st.link_button("Open vragenlijst", url)
    st.caption("Werkt de knop niet? Kopieer dan de onderstaande link en plak deze in uw browser.")
    st.code(url, language="text")

def run_experiment():
    _init_participant_state()
    _progress()

    step = st.session_state.exp_step
    if step == 1:
        step_1_consent()
    elif step == 2:
        step_2_assignment()
    elif step == 3:
        step_3_scenario()
    else:
        step_4_redirect()
