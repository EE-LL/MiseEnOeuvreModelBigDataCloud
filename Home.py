import streamlit as st

# Configuration de la page
st.set_page_config(
    page_title="Accueil | Risk Banking",
    page_icon="🏦",
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

# Titre principal
st.title("🏦 Bienvenue sur la plateforme Risk Banking")

# Description
st.markdown("""
Bienvenue sur **Risk Banking**, la plateforme intelligente d'analyse de risque crédit.  
Notre application vous permet de :

- 🔍 **Analyser le risque de défaut d’un client individuel**
- 📊 **Explorer les tendances globales de défaut dans le portefeuille client**
- 🧠 **Prendre des décisions éclairées grâce à la puissance de Databricks et du Machine Learning**

---

### 🚀 Accès rapide

#### 🧑‍💼 Analyse d'un client spécifique
Évaluez en quelques secondes le profil de risque d’un client grâce à son ID.

➡️ Naviguez dans le menu à gauche → *Analyse Client*

#### 📈 Analyse thématique des données
Visualisez des graphiques interactifs et appliquez des filtres personnalisés pour explorer les facteurs de risque.

➡️ Naviguez dans le menu à gauche → *Analyse Données Crédit*

---

### 🛠️ Technologies utilisées

- **Streamlit** pour l’interface
- **Flask** comme middleware backend
- **Databricks (Spark)** pour l’exécution des modèles
- **MongoDB Atlas** pour le stockage des données clients
- **Azure Application Insights** pour le monitoring

---

""")

# Image illustrative
st.image("assets/risk_banking.jpg", width=500)

# Footer
st.markdown("---")
st.caption("📅 Dernière mise à jour : Juillet 2025 | © Sup de Vinci - M2DATA | Application pédagogique")
