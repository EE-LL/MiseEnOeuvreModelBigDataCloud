import streamlit as st
import pymongo
import plotly.graph_objects as go
from dotenv import load_dotenv
import os
import requests

# --- 1. CONFIGURATION ET CONNEXION ---

# Charger les variables d'environnement
load_dotenv()

# Configuration de la page
st.set_page_config(
    page_title="Risk Banking | Prédiction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fonction pour initialiser la connexion à MongoDB (mise en cache)
@st.cache_resource
def init_connection():
    """Initialise et retourne un client MongoDB."""
    try:
        uri = os.environ.get("MONGODB_URI") 
        if not uri:
            st.error("L'URI de MongoDB n'est pas configurée. Vérifiez votre fichier .env.")
            return None
        client = pymongo.MongoClient(uri)
        return client
    except Exception as e:
        st.error(f"Erreur de connexion à MongoDB : {e}")
        return None

# Initialiser la connexion au démarrage
mongo_client = init_connection()


#  Fonction de récupération de données
def get_client_personal_data(client, client_id):
    """Récupère les données personnelles depuis MongoDB."""
    if client is None: return None
    try:
        db = client.default_risk
        collection = db.users_data
        query = {"SK_CURR_ID": int(client_id)}
        user_data = collection.find_one(query)
        if user_data is None:
            query = {"SK_CURR_ID": str(client_id)}
            user_data = collection.find_one(query)
        return user_data
    except Exception as e:
        st.warning(f"Impossible de récupérer les données personnelles : {str(e)}")
        return None
    
#  Fonction de récupération de predictions
def predict_default_risk(client_id: str):
    """Appelle l'endpoint /predict_default de l'API Flask pour obtenir une prédiction."""
    API_URL = os.environ.get('FLASK_API_URL')
    if not API_URL:
        st.error("L'URL de l'API Flask n'est pas configurée. Vérifiez votre fichier .env.")
        return {"error": "URL de l'API non configurée"}
        
    endpoint_url = f"{API_URL}/predict_default"
    params = {"client_id": client_id}
    try:
        response = requests.get(endpoint_url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur de communication avec l'API de prédiction : {e}")
        return {"error": "Impossible de joindre l'API de prédiction."}
    
# Fonction de feedback 
def send_feedback_to_api(message: str, client_id: str):
    """Envoie un feedback (erreur) à l'API pour logging sur App Insights."""
    API_URL = os.environ.get('FLASK_API_URL')
    if not API_URL:
        st.error("L'URL de l'API Flask n'est pas configurée.")
        return False
        
    endpoint_url = f"{API_URL}/feedback"
    payload = {
        "message": message,
        "is_positive": False, # C'est un signalement d'erreur
        "custom_dimensions": {
            "page": "Prédiction Client",
            "client_id_context": client_id
        }
    }
    try:
        response = requests.post(endpoint_url, json=payload)
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur lors de l'envoi du feedback : {e}")
        return False


# --- INTERFACE UTILISATEUR (UI) ---

# --- Barre Latérale ---
with st.sidebar:
    # st.markdown("### 🏦 Risk Banking")
    # st.markdown("Plateforme d'analyse de risque")
    # st.markdown("---")
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image("assets/logo.png", width=120) 
    
    st.markdown("<h1 style='text-align: center; color: #004488;'>Risk Banking</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; margin-top: -10px;'>Analyses & Intelligence Décisionnelle</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.header("Rechercher un client")
    client_id_input = st.text_input("ID Client", "118893")
    analyze_button = st.button("🔍 Analyser le risque", type="primary")
    
    # st.image("assets/logo.png", width=80) 
    # st.markdown("<h1 style='text-align: center; margin-top: -20px; color: #004488;'>Risk Banking</h1>", unsafe_allow_html=True)
    # st.markdown("<p style='text-align: center; margin-top: -10px;'>Analyses & Intelligence Décisionnelle</p>", unsafe_allow_html=True)
    # st.markdown("---")
    # st.header("🔍 Rechercher un client")
    # client_id_input = st.text_input("ID Client", "118893")
    # analyze_button = st.button("Analyser le risque", type="primary")
    
    # ajout feedback
    st.markdown("---")
    with st.expander("❓Feedback"):
        feedback_text = st.text_area(
            "Décrivez l'erreur ou le comportement inattendu :", 
            key="feedback_area",
            placeholder="Ex: Le score de risque me semble incorrect pour ce client..."
        )
        if st.button("Envoyer le signalement"):
            if feedback_text:
                success = send_feedback_to_api(feedback_text, client_id_input)
                if success:
                    st.success("Merci ! Votre signalement a bien été envoyé.")
                # L'échec est déjà géré par une st.error dans la fonction
            else:
                st.warning("Veuillez décrire le problème avant d'envoyer.")


# --- Page Principale ---
st.title("🔮 Prédiction de Défaut Client")
st.info("Cette page permet d'estimer la probabilité de défaut d'un client. Entrez un ID pour obtenir une analyse complète.", icon="ℹ️")

# --- Style CSS (défini une seule fois) ---
st.markdown("""
<style>
.profile-image {
    width: 150px; height: 150px; border-radius: 50%; object-fit: cover;
    display: block; margin-left: auto; margin-right: auto;
}
.profile-name {
    font-size: 20px; font-weight: bold; text-align: center; margin-top: 15px;
}
.profile-id {
    font-size: 14px; color: grey; text-align: center; margin-top: 5px;
}
.recommendation-box {
    background-color: #f0f2f6; border: 1px solid #dcdcdc;
    border-radius: 10px; padding: 25px; margin-top: 20px;
}

/* --- NOUVEAUX STYLES POUR LE STATUT DE RISQUE --- */
.status-tag {
    padding: 5px 15px;
    border-radius: 15px;
    color: white;
    font-weight: bold;
    text-align: center;
    display: inline-block;
}
.status-low { background-color: #28a745; } /* Vert */
.status-medium { background-color: #ffc107; } /* Orange */
.status-high { background-color: #dc3545; } /* Rouge */
</style>
</style>
""", unsafe_allow_html=True)


# --- LOGIQUE D'AFFICHAGE CONDITIONNELLE ---
# Ce bloc ne s'exécute QUE si le bouton est cliqué et qu'un ID est saisi
if analyze_button and client_id_input:
    with st.spinner("Analyse en cours..."):
        # On récupère TOUTES les données d'abord
        personal_data = get_client_personal_data(mongo_client, client_id_input)
        prediction_result = predict_default_risk(client_id_input)

    # On vérifie qu'on a bien reçu toutes les données nécessaires
    if personal_data and prediction_result and "error" not in prediction_result:
        
        # --- Affichage du Profil Client ---
        st.markdown(f'<img src="{personal_data.get("PhotoURL", "")}" class="profile-image">', unsafe_allow_html=True)
        st.markdown(f'<p class="profile-name">{personal_data.get("FirstName", "")} {personal_data.get("LastName", "")}</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="profile-id">ID Client: {personal_data.get("SK_CURR_ID", "")}</p>', unsafe_allow_html=True)

        # Ajoute une ligne de séparation
        st.markdown("---")

        # --- Affichage de l'Analyse de Risque ---
        st.subheader("📊 Résultat de l'analyse de risque")
        
        risk_score = prediction_result['prediction']['risk_score']
        score_percent = risk_score * 100
        
        def render_risk_gauge(score_percent):
            colors = ["green", "orange", "red"]
            thresholds = [40, 70, 100]

            # Crée les segments empilés comme une jauge horizontale
            fig = go.Figure()
            start = 0
            for i, end in enumerate(thresholds):
                fig.add_trace(go.Bar(
                    x=[end - start],
                    y=[""],
                    orientation='h',
                    marker=dict(color=colors[i]),
                    showlegend=False,
                    hoverinfo="skip",
                    width=0.2  # ➜ rend la jauge très fine
                ))
                start = end

            # Ajoute UN seul curseur rond blanc avec contour noir
            fig.add_trace(go.Scatter(
                x=[score_percent],
                y=[""],
                mode="markers",
                marker=dict(
                    size=14,
                    color="white",
                    line=dict(color="black", width=2)
                ),
                hovertemplate=f"{score_percent:.1f}%<extra></extra>",
                showlegend=False
            ))

            # Met en forme la figure
            fig.update_layout(
                barmode='stack',
                height=50,  # compact
                width=550,  # à adapter selon la place
                margin=dict(l=0, r=0, t=10, b=10),
                xaxis=dict(range=[0, 100], visible=False),
                yaxis=dict(visible=False),
                plot_bgcolor="rgba(0,0,0,0)",  # fond transparent
                paper_bgcolor="rgba(0,0,0,0)"  # fond transparent
            )

            return fig
        
        
        # --- Affichage de la jauge avec curseur rond blanc ---
        label_col1, spacer_left, gauge_col, spacer_right, label_col2, score_col = st.columns([1, 1, 5, 1, 1, 1.5])
        
        with label_col1:
            st.markdown("<p style='text-align: right;'>Faible</p>", unsafe_allow_html=True)
        
        with gauge_col:
            fig = render_risk_gauge(score_percent)
            st.plotly_chart(fig, use_container_width=False)

        with label_col2:
            st.markdown("<p style='text-align: left;'>Élevé</p>", unsafe_allow_html=True)

        with score_col:
            st.markdown(f"<h3 style='text-align: left; color: orange;'>{score_percent:.1f}%</h3>", unsafe_allow_html=True)
            
        # --- Affichage de Recommandation ---
        with st.container():
            st.subheader("Recommandation")
            st.write(prediction_result['recommendation']['explanation'])
            st.markdown("---")
            
            income = prediction_result['client_info'].get('income', 0)
            credit_amount = prediction_result['client_info'].get('credit_amount', 0)
            ratio = round(int(credit_amount)/int(income), 2)
            
            rec_col1, rec_col2, rec_col3 = st.columns(3)
            with rec_col1: st.metric("Demande", f"{credit_amount} €".replace(",", " "))
            with rec_col2: st.metric("Ratio crédit/revenu", f"{ratio}")
            with rec_col3: st.metric("Date d'analyse", f"{prediction_result['metadata'].get('analysis_date', 0)}")
            st.markdown('</div>', unsafe_allow_html=True)

        # --- Le statut de risque ---
        st.text("")
        
        # On détermine le texte ET la classe CSS en fonction du score
        if risk_score < 0.4:
            risk_level_text = "Risque Faible"
            status_class = "status-low"
        elif risk_score < 0.7:
            risk_level_text = "Risque Moyen"
            status_class = "status-medium"
        else:
            risk_level_text = "Risque Élevé"
            status_class = "status-high"
        
        # On utilise des colonnes pour centrer le tag
        _, col_button, _ = st.columns([2, 1, 2])
        with col_button:
            # On affiche le tag en utilisant st.markdown et notre style CSS
            st.markdown(f'<div class="status-tag {status_class}">{risk_level_text}</div>', unsafe_allow_html=True)


    else:
        # Message d'erreur si les données n'ont pas pu être récupérées
        st.error(f"Impossible de récupérer les informations complètes pour le client ID : {client_id_input}. Veuillez vérifier l'identifiant et que l'API est bien lancée.")