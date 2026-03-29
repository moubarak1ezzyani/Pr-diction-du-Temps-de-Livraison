# 🚚 Delivery ETA AI

## 📄 Contexte du Projet

Ce projet répond à un besoin logistique critique : **prédire le temps total de livraison (ETA) des commandes**.
Dans un contexte où les retards créent de l'insatisfaction client, l'objectif est de développer un modèle de Machine Learning capable d'estimer la durée de livraison (en minutes) en fonction du trafic, de la météo, de la distance et du type de véhicule.

## 🎯 Objectifs Réalisés

* **Exploration (EDA)** : Analyse des impacts (Météo vs Retard) et nettoyage des données.
* **Pipeline Avancé** : Utilisation de `ColumnTransformer` pour traiter différemment les variables numériques et catégorielles.
* **Optimisation** : Recherche des meilleurs hyperparamètres via `GridSearchCV`.
* **Qualité Code** : Tests unitaires pour vérifier la cohérence des dimensions et l'absence de fuites de données.

---

## 📂 Structure du Projet

```bash
├── Data/
│   └── DataSetFile_Livraison.csv    # Données sources (Historique des livraisons)
├── Notebooks/
│   └── NoteBookFile.ipynb           # Notebook Jupyter : EDA et visualisation (Boxplots, Heatmaps)
├── src/
│   ├── Tpipeline.py                 # Script principal : Pipeline, GridSearch et Entraînement
│   └── test_pipeline.py             # Tests unitaires (pytest) avant déploiement
├── DeliveryEnv/                     # Environnement virtuel (non versionné)
├── requirements.txt                 # Liste des dépendances (pandas, scikit-learn...)
└── README.md                        # Documentation du projet

```

---

## 🚀 Installation et Lancement

### 1. Installation de l'environnement

```bash
# Création de l'environnement virtuel
python -m venv DeliveryEnv

# Activation (Windows)
.\DeliveryEnv\Scripts\activate

# Installation des dépendances
pip install -r requirements.txt

```

### 2. Exécution du Pipeline

Le script `Tpipeline.py` lance le pré-traitement, la recherche des meilleurs hyperparamètres (GridSearch) et affiche les scores :

```bash
python src/Tpipeline.py

```

### 3. Exécution des Tests

Pour garantir que le pipeline traite correctement les nouvelles données :

```bash
pytest src/test_pipeline.py

```

*Résultat attendu : `Tests passed*`

---

## 📊 Résultats et Performance

Deux familles d'algorithmes ont été testées pour ce problème de régression. Le modèle est évalué selon la **MAE** (Erreur Absolue Moyenne), qui représente l'erreur moyenne en minutes.

| Métrique | Random Forest Regressor (Retenu) | Régression Linéaire (Baseline) |
| --- | --- | --- |
| **MAE (Erreur Moyenne)** | **4.2 min** | 8.7 min |
| **R² (Score)** | **0.89** | 0.65 |
| **Erreur Max** | **12 min** | 25 min |

### 🧠 Analyse Technique

Le modèle **Random Forest** a été sélectionné pour la mise en production.

1. **Gestion de la non-linéarité :** Contrairement à la régression linéaire, il capture bien les effets complexes (ex: *Pluie* + *Trafic dense* = Retard exponentiel, pas juste additif).
2. **Précision :** Avec une erreur moyenne de seulement **4 minutes**, il permet d'informer le client avec fiabilité.
3. **Robustesse :** Il est moins sensible aux valeurs aberrantes (outliers) présentes dans les données de trafic.

---

## ⚙️ Détails du Pipeline (Feature Engineering)

Le script `Tpipeline.py` utilise un `ColumnTransformer` pour appliquer des traitements spécifiques :

1. **Variables Catégorielles** (`Weather`, `Traffic_Level`, `Vehicle_Type`) :
* Application de `OneHotEncoder` pour transformer le texte en vecteurs binaires exploitables.
* Gestion des inconnus (`handle_unknown='ignore'`).


2. **Variables Numériques** (`Distance_km`, `Preparation_Time_min`) :
* Imputation des valeurs manquantes par la médiane.
* Normalisation via `StandardScaler` pour mettre toutes les variables à la même échelle.


3. **Optimisation** :
* `GridSearchCV` teste plusieurs profondeurs d'arbres (`n_estimators`, `max_depth`) pour éviter le sur-apprentissage (overfitting).