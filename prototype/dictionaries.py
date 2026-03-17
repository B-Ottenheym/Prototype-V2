variable_descriptions = {
    "payment_reliability": "Mate waarin de opdrachtgever betalingen tijdig en betrouwbaar verricht (schaal 1–5).",
    "decision_making_structure": "Aantal beslissings- en goedkeuringslagen binnen de organisatie van de opdrachtgever (1–10).",
    "change_order_freq": "Hoe vaak contractwijzigingen historisch voorkomen bij de opdrachtgever, als percentage van contracten.",
    "contractor_experience": "Aantal jaren relevante ervaring van de aannemer in vergelijkbare projecten.",
    "num_subcontractors": "Het aantal onderaannemers dat bij het project betrokken is.",
    "planning_detail": "Hoe gedetailleerd en uitgewerkt de planning van de aannemer is (% voltooiing van de baselineplanning).",
    "design_maturity": "De mate waarin het ontwerp volledig is afgerond en bevroren vóór de start van de uitvoering (%).",
    "pre_design_data_completeness": "De volledigheid van beschikbare pre‑ontwerpgegevens (%).",
    "design_team_experience_years": "Gemiddeld aantal jaren ervaring binnen het ontwerpteam.",
    "use_of_bim": "Of het project BIM‑technologie gebruikt (0 = nee, 1 = ja).",
    "labour_shortage": "Het verwachte arbeidstekort (% onder het benodigde niveau).",
    "experienced_labour": "Het aandeel ervaren of gecertificeerde arbeidskrachten (%).",
    "consultant_experience": "Aantal jaren ervaring van de betrokken adviseur(s).",
    "consultant_availability": "Beschikbaarheid van de adviseur(s), rekening houdend met hun werkbelasting (%).",
    "consultant_prior_collaboration": "Of de adviseur eerder met deze aannemer heeft samengewerkt (0 = nee, 1 = ja).",
    "material_supply_reliability": "Hoe betrouwbaar de levering van materialen is (schaal 1–5).",
    "expected_inflation": "De verwachte inflatie of kostenstijging (%).",
    "weather_risk": "Het percentage werkdagen dat waarschijnlijk wordt beïnvloed door slechte weersomstandigheden.",
    "subsurface_risk": "Risico op problemen door ondergrondse omstandigheden (schaal 1–5).",
    "permitting_complexity": "De complexiteit van vergunningprocedures (schaal 1–5).",
    "project_size_m2": "De totale omvang van het project in vierkante meters.",
    "project_type": "Het type project (bijv. Woningbouw, Commercieel, Industrieel, enz.).",
    "contract_value_million": "De totale contractwaarde in miljoenen euro’s.",
    "contract_award_method": "Methode waarop het contract is gegund (laagste bieding, onderhandeling, kwaliteitsgericht).",
    "contract_type": "Het type contract dat is toegepast (bijv. Lump‑sum, Design‑build, Cost‑plus).",
    "equipment_availability": "De mate waarin benodigd materieel beschikbaar is (ratio 0–1).",
    "average_equipment_age": "De gemiddelde leeftijd van het ingezette materieel (jaren)."
}

variable_labels = {
    "payment_reliability": "Betrouwbaarheid van betalingen",
    "decision_making_structure": "Besluitvormingsstructuur",
    "change_order_freq": "Historische frequentie van contractwijzigingen",
    "contractor_experience": "Ervaring van aannemer",
    "num_subcontractors": "Aantal onderaannemers",
    "planning_detail": "Detailniveau van planning",
    "design_maturity": "Ontwerpgereedheid",
    "pre_design_data_completeness": "Volledigheid van pre-ontwerpgegevens",
    "design_team_experience_years": "Ervaring ontwerpteam",
    "labour_shortage": "Arbeidstekort",
    "experienced_labour": "Aandeel ervaren arbeid",
    "consultant_experience": "Ervaring van adviseur",
    "consultant_availability": "Beschikbaarheid van adviseur",
    "material_supply_reliability": "Betrouwbaarheid van materiaallevering",
    "expected_inflation": "Verwachte kostenstijging",
    "weather_risk": "Weerrisico op vertraging",
    "subsurface_risk": "Ondergrondrisico",
    "permitting_complexity": "Vergunningstechnische complexiteit",
    "equipment_availability": "Beschikbaarheid van materieel",
    "average_equipment_age": "Gemiddelde leeftijd van materieel",
    "project_size_m2": "Projectomvang",
    "contract_value_million": "Contractwaarde",
    "planned_duration_days": "Geplande doorlooptijd",
    "project_type": "Projecttype",
    "contract_award_method": "Gunningsmethode",
    "contract_type": "Contracttype",
    "use_of_bim": "Gebruik van BIM",
    "consultant_prior_collaboration": "Eerdere samenwerking met consultant",
}

variable_labels2 = {
    "payment_reliability": "Betrouwbaarheid van betalingen",
    "decision_making_structure": "Besluitvormingsstructuur",
    "change_order_freq": "Historische frequentie van contractwijzigingen",
    "contractor_experience": "Ervaring van aannemer",
    "num_subcontractors": "Aantal onderaannemers",
    "planning_detail": "Detailniveau van planning",
    "design_maturity": "Ontwerpgereedheid",
    "pre_design_data_completeness": "Volledigheid van pre-ontwerpgegevens",
    "design_team_experience_years": "Ervaring ontwerpteam",
    "labour_shortage": "Arbeidstekort",
    "experienced_labour": "Aandeel ervaren arbeid",
    "consultant_experience": "Ervaring van adviseur",
    "consultant_availability": "Beschikbaarheid van adviseur",
    "material_supply_reliability": "Betrouwbaarheid van materiaallevering",
    "expected_inflation": "Verwachte kostenstijging",
    "weather_risk": "Weerrisico op vertraging",
    "subsurface_risk": "Ondergrondrisico",
    "permitting_complexity": "Vergunningstechnische complexiteit",
    "equipment_availability": "Beschikbaarheid van materieel",
    "average_equipment_age": "Gemiddelde leeftijd van materieel",
    "project_size_m2": "Projectomvang",
    "contract_value_million": "Contractwaarde",
    "planned_duration_days": "Geplande doorlooptijd",
    "project_type": "Projecttype",
    "contract_award_method": "Gunningsmethode",
    "contract_type": "Contracttype",
    "use_of_bim": "Gebruik van BIM",
    "consultant_prior_collaboration": "Eerdere samenwerking met consultant",
    "use_of_bim_1": "Gebruik van BIM",
    "consultant_prior_collaboration_1": "Eerdere samenwerking met consultant"
}

variable_groups = {
    "Eigenaar-gerelateerde factoren": [
        "payment_reliability", "decision_making_structure", "change_order_freq"
    ],
    "Aannemer gerelateerde factoren": [
        "contractor_experience", "num_subcontractors", "planning_detail"
    ],
    "Ontwerpteam-gerelateerde factoren": [
        "design_maturity", "pre_design_data_completeness",
        "design_team_experience_years", "use_of_bim"
    ],
    "Arbeidsgerelateerde factoren": [
        "labour_shortage", "experienced_labour"
    ],
    "Adviseurgerelateerde factoren": [
        "consultant_experience", "consultant_availability", "consultant_prior_collaboration"
    ],
    "Materiaal-gerelateerde factoren": [
        "material_supply_reliability", "expected_inflation"
    ],
    "Externe factoren": [
        "weather_risk", "subsurface_risk", "permitting_complexity"
    ],
    "Project-gerelateerde factoren": [
        "project_size_m2", "contract_value_million", "planned_duration_days",
        "project_type", "contract_award_method", "contract_type"
    ],
    "Materieel-gerelateerde factoren": [
        "equipment_availability", "average_equipment_age"
    ]
}