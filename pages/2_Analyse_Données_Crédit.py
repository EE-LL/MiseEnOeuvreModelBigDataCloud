import streamlit as st
import os
import requests
from typing import Dict, Any, Optional, List
import json


# CONFIGURATION DE LA PAGE STREAMLIT

st.set_page_config(
    page_title="Analyse de Données | Risk Banking",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Barre Latérale ---
with st.sidebar:
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image("assets/logo.png", width=120) 
    
    st.markdown("<h1 style='text-align: center; color: #004488;'>Risk Banking</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; margin-top: -10px;'>Analyses & Intelligence Décisionnelle</p>", unsafe_allow_html=True)

API_BASE_URL = os.getenv("FLASK_API_URL", "http://127.0.0.1:5001")

# IMPORTS ET FONCTIONS UTILITAIRES (CLIENT-SIDE)
# Dictionnaire des analyses
ANALYSES = {
    1: {"title": "Âge client et défaut", "icon": "👶👴", "description": "Analyse de l'âge des clients et son impact sur les défauts de paiement."},
    2: {"title": "Ratio crédit/revenu", "icon": "💰💸", "description": "Analyse du ratio entre le montant du crédit et les revenus du client."},
    3: {"title": "Type de revenu", "icon": "💼💵", "description": "Impact du type de revenu (salarié, retraité...) sur le risque de défaut."},
    4: {"title": "Ancienneté d'emploi", "icon": "📅👔", "description": "Relation entre la stabilité de l'emploi et le taux de défaut."},
    5: {"title": "Demandes de crédit", "icon": "📊📄", "description": "Influence du nombre de demandes de crédit récentes sur le risque."},
    6: {"title": "Contrat et propriété", "icon": "🏠📝", "description": "Analyse croisée du type de contrat et de la possession d'un bien immobilier."},
    7: {"title": "Famille et enfants", "icon": "👨‍👩‍👧‍👦👶", "description": "Impact de la situation familiale et du nombre d'enfants sur le crédit."},
    8: {"title": "Jour de demande", "icon": "🗓️🔄", "description": "Variation du risque selon le jour de la semaine où le crédit est demandé."}
}

# Fonction pour appeler l'API
@st.cache_data(ttl=600)
def load_credit_analysis_data(
    analysis_type: int,
    min_credit: Optional[int],
    max_credit: Optional[int],
    min_income: Optional[int],
    family_statuses: Optional[List[str]], # Nouveau paramètre
    income_types: Optional[List[str]],    # Nouveau paramètre
    realty_status: Optional[str]          # Nouveau paramètre
) -> Dict[str, Any]:
    """Appelle l'API Flask en incluant les filtres avancés."""
    if not API_BASE_URL:
        return {"status": "error", "content": {"error": "FLASK_API_URL non défini"}, "type": "json"}

    params = {
        "job_id": "streamlit_run",
        "analysis_type": analysis_type,
        "min_credit": min_credit,
        "max_credit": max_credit,
        "min_income": min_income,
        # On envoie les listes comme des chaînes JSON
        "family_statuses": json.dumps(family_statuses) if family_statuses else None,
        "income_types": json.dumps(income_types) if income_types else None,
        "realty_status": realty_status
    }

    try:
        response = requests.get(f"{API_BASE_URL}/get_dataviz", params=params, timeout=300)
        response.raise_for_status()
        return {"status": "success", "content": response.text, "type": "html"}
    except requests.exceptions.RequestException as e:
        return {"status": "error", "content": {"error": str(e)}, "type": "json"}
    
    
@st.cache_data(ttl=3600) # On cache pour 1h
def load_filter_options() -> Dict[str, Any]:
    """Charge les options pour les filtres depuis l'API."""
    try:
        api_url = os.getenv("FLASK_API_URL", "http://127.0.0.1:5001")
        response = requests.get(f"{api_url}/get_filter_options")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Impossible de charger les options de filtre: {e}")
        # Retourne des listes vides en cas d'erreur
        return {"family_status_options": [], "income_type_options": []}
    
# Chargement des options dynamiquement
filter_options = load_filter_options()

with st.sidebar.expander("Filtres avancés"):
    
    # Utiliser les options chargées depuis l'API
    selected_family_statuses = st.multiselect(
        "Statut familial",
        options=filter_options.get("family_status_options", []),
        default=filter_options.get("family_status_options", [])
    )

    selected_income_types = st.multiselect(
        "Type de revenu",
        options=filter_options.get("income_type_options", []),
        default=filter_options.get("income_type_options", [])
    )

    owns_realty_choice = st.radio(
        "Propriétaire d'un bien immobilier",
        options=["Tous", "Oui", "Non"],
        horizontal=True
    )
    

# Fonctions d'affichage
def display_plotly_chart(html_content):
    st.components.v1.html(html_content, height=715, scrolling=True)

def display_key_metrics(analysis_type):
    st.subheader("Indicateurs Clés")
    col1, col2, col3 = st.columns(3)
    # Implémentation simplifiée pour l'exemple
    if analysis_type == 1:
        with col1: st.metric(label="Groupe le + à risque", value="> 55 ans", delta="9.87%", delta_color="inverse")
        with col2: st.metric(label="Groupe le - à risque", value="35-45 ans", delta="7.33%", delta_color="normal")
        with col3: st.metric(label="Écart de risque", value="2.54 pts")
    else:
        st.info("Métriques non configurées pour ce type d'analyse.")

def display_recommendations(analysis_type):
    st.subheader("Recommandations et Actions")
    if analysis_type == 1:
        st.markdown("##### 🎯 Recommandations Opérationnelles")
        st.markdown("- **Renforcer les vérifications** pour les dossiers des moins de 25 ans.")
        st.markdown("##### 🚀 Recommandations Stratégiques")
        st.markdown("- **Développer des produits de crédit spécifiques** pour les jeunes actifs.")
    else:
        st.info("Recommandations non configurées pour ce type d'analyse.")


# BARRE LATÉRALE (SIDEBAR)

st.sidebar.header("THÉMATIQUE D'ANALYSE")
selected_analysis_id = st.sidebar.radio(
    label="Choisissez un type d'analyse :",
    options=ANALYSES.keys(),
    format_func=lambda x: f"{ANALYSES[x]['icon']} {ANALYSES[x]['title']}"
)
st.sidebar.info(ANALYSES[selected_analysis_id]['description'])

st.sidebar.markdown("---")
st.sidebar.header("FILTRES D'ANALYSE")
credit_range = st.sidebar.slider("Montant du crédit (€)", 0, 2000000, (50000, 750000), 10000)
min_credit_val, max_credit_val = credit_range

income_range = st.sidebar.slider("Revenu annuel (€)", 0, 1000000, (25000, 300000), 5000)
min_income_val, max_income_val = income_range # Note: la fonction API ne prend qu'un min_income

# Bouton pour lancer l'analyse, placé à la fin de la sidebar
analyze_button = st.sidebar.button("📊 Générer l'analyse")


# CORPS PRINCIPAL DE LA PAGE

st.title("📈 Analyse de Données Crédit")
st.markdown("Explorez les tendances et les facteurs de risque au sein de notre portefeuille de crédits.")

# Si l'utilisateur clique sur le bouton "Générer l'analyse"
if analyze_button:
    with st.spinner("Chargement de l'analyse en cours... Veuillez patienter."):
        # On appelle la fonction en lui passant TOUS les paramètres
        analysis_result = load_credit_analysis_data(
            analysis_type=selected_analysis_id,
            min_credit=min_credit_val,
            max_credit=max_credit_val,
            min_income=min_income_val,
            # Ajout des arguments manquants
            family_statuses=selected_family_statuses,
            income_types=selected_income_types,
            realty_status=owns_realty_choice
        )

    # Une fois le chargement terminé, on affiche les résultats
    st.header(f"Résultats pour : {ANALYSES[selected_analysis_id]['title']}")

    if analysis_result['status'] == 'success':
        # Afficher le graphique HTML reçu de l'API
        display_plotly_chart(analysis_result['content'])
        
        st.markdown("---")
        
        # Afficher les métriques et les recommandations
        display_key_metrics(selected_analysis_id)
        display_recommendations(selected_analysis_id)
        
    else:
        # En cas d'erreur, on affiche le message
        st.error("Une erreur est survenue lors de la récupération des données :")
        st.json(analysis_result.get('content', {'error': 'Erreur inconnue'}))

else:
    # Message d'accueil affiché par défaut avant de lancer une analyse
    st.info("Veuillez sélectionner un type d'analyse et cliquer sur 'Générer l'analyse' pour commencer.")