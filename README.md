# 🏦 Risk Banking - Plateforme d'Analyse de Risque Crédit

---

Ce projet est une application web complète conçue pour aider les conseillers bancaires et les analystes de risque à évaluer et comprendre le risque de défaut de paiement des clients. L'application se compose d'une interface utilisateur interactive construite avec Streamlit, qui communique avec une API Flask servant de middleware pour orchestrer des calculs complexes sur un cluster Databricks.

## 🌟 Aperçu de l'application
L'application offre deux fonctionnalités principales :

**Prédiction de Défaut Client** : Permet d'obtenir une évaluation de risque instantanée pour un client spécifique, incluant un score de probabilité, une recommandation claire (Acceptation, Refus, etc.), et les facteurs influençant la décision.

**Analyse de Données Crédit** : Offre un tableau de bord interactif pour explorer les tendances du portefeuille de crédits à travers 8 analyses thématiques (par âge, par revenu, etc.) avec des filtres dynamiques.

## 🏗️ Architecture Technique
Le projet est basé sur une architecture microservices moderne, conçue pour séparer les responsabilités et permettre une scalabilité efficace.

```mermaid
graph TD
    A[Utilisateur (Conseiller/Analyste)] -- Interagit avec --> B(Frontend: Streamlit);
    B -- Requêtes HTTP --> C{API Backend: Flask};
    C -- Lance des jobs --> D(Calculs Big Data: Databricks);
    C -- Récupère les logs --> E(Monitoring: Azure App Insights);
    C -- Récupère les données personnelles --> F(Base de données: MongoDB);
    D -- Lit les données de crédit --> G[Stockage: DBFS];
```

**Frontend** (client_prediction.py, data_analysis.py) : Construit avec **Streamlit** pour une interface utilisateur réactive et facile à développer.

**Backend / Middleware** (app.py) : Une **API Flask** qui sert de pont. Elle reçoit les requêtes de l'interface, lance les jobs sur Databricks, et formate les réponses.

**Moteur de Calcul** : Un cluster **Azure Databricks** exécute des scripts **PySpark** pour les analyses de données et les prédictions de modèles de Machine Learning.

**Stockage de Données** : Les données de crédit sont stockées sur **Databricks File System (DBFS)** et les données personnelles des clients sur **MongoDB Atlas**.

**Monitoring & Logging** : Les feedbacks sont centralisés dans **Azure Application Insights** pour un suivi en temps réel.

## ✨ Fonctionnalités Clés

### Page de Prédiction de Défaut Client
- [x] Recherche de client par ID.

- [x] Affichage des informations personnelles du client (nom, photo) depuis MongoDB.

- [x] Jauge de risque visuelle (de 0 à 100%).

- [x] Recommandation claire et plan d'action détaillé pour le conseiller.

- [x] Système de feedback pour signaler les prédictions incorrectes à App Insights.

### Page d'Analyse de Données Crédit
- [x] 8 analyses thématiques interactives (Âge, Revenu, Ancienneté, etc.).

- [x] Filtres dynamiques pour affiner les analyses (montant, revenu, statut familial...).

- [x] Visualisations de données interactives générées avec Plotly.

- [x] Indicateurs clés (KPIs) résumant les principaux insights.

- [x] Recommandations opérationnelles et stratégiques basées sur les tendances observées.

## 🛠️ Stack Technique
**Frontend**: Streamlit

**Backend** : Flask

**Calcul Big Data** : Azure Databricks, Apache Spark (PySpark)

**Base de Données** : MongoDB

**Visualisation** : Plotly

**Monitoring** : Azure Application Insights

**Langage** : Python 3.9+

## 🚀 Installation et Lancement

Suivez ces étapes pour lancer le projet en local.

### Prérequis
- Python 3.9 ou supérieur
- Un compte Azure avec un cluster Databricks configuré.
- Un compte MongoDB Atlas avec les données clients.
- Les identifiants pour Databricks, MongoDB et App Insights.

### 1. Cloner le Dépôt
```bash
git clone [https://github.com/EE-LL/MiseEnOeuvreModelBigDataCloud.git](https://github.com/EE-LL/MiseEnOeuvreModelBigDataCloud.git)
```

### 2. Créer un Environnement Virtuel
```bash
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate
```

### 3. Installer les Dépendances
```bash
pip install -r requirements.txt
```

### 4. Configurer les Variables d'Environnement
Créez un fichier `.env` à la racine du projet et remplissez-le avec vos informations :

```env
# URL de l'API Flask (pour la communication Streamlit -> Flask)
FLASK_API_URL=[http://127.0.0.1:5001](http://127.0.0.1:5001)

# Identifiants Databricks
DATABRICKS_HOST=[https://adb-xxxxxxxxxxxxxxxx.xx.azuredatabricks.net](https://adb-xxxxxxxxxxxxxxxx.xx.azuredatabricks.net)
DATABRICKS_TOKEN=dapixxxxxxxxxxxxxxxxxxxxxxxx
CLUSTER_ID=xxxx-xxxxxx-xxxxxxxx

# Identifiants MongoDB
MONGODB_URI="mongodb+srv://user:password@cluster..."

# Identifiants Azure Application Insights
CONNECTION_STRING="InstrumentationKey=..."
```

### 5. Lancer l'API Flask
Ouvrez un premier terminal et lancez le serveur backend :

```bash
python app.py
```
Le serveur devrait démarrer sur `http://127.0.0.1:5001`.

### 6. Lancer l'Application Streamlit
Ouvrez un second terminal et lancez l'interface utilisateur. Vous pouvez choisir quelle page lancer :

**Pour la page de prédiction :**
```bash
streamlit run client_prediction.py
```

**Pour la page d'analyse de données :**
```bash
streamlit run data_analysis.py
```
L'application s'ouvrira automatiquement dans votre navigateur à l'adresse `http://localhost:8501`.

## 📖 Structure du Projet
```
.
├── Home.py                 # Page d'accueil de l'application Streamlit
├── app.py                  # API Backend (Flask)
├── pages/
│   ├── 2_Prédiction_Client.py    # Page Streamlit pour la prédiction
│   └── 3_Analyse_de_Données.py # Page Streamlit pour l'analyse
├── assets/                 # Images
│   └── logo.png           
│   └── risk_banking.png         
├── requirements.txt        # Dépendances Python
├── .env                    # Fichier pour les variables d'environnement
└── README.md               # Ce fichier
