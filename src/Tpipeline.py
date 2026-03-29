import unittest
import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, make_scorer


# --- loading data
def loading_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    csv_path = os.path.join(project_root, "Data", "DataSetFile_Livraison.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Le fichier CSV est introuvable  : {csv_path}")

    data_file = pd.read_csv(csv_path)
    print(f"Dimensions : {data_file.shape}")
    return data_file


# --- Tests Unitaires Simples
def run_unit_tests(data_file_df):
    print("\n Exécution des Tests Unitaires (Phase Pré-Pipeline) ---")

    # Test 1: Vérifier la présence de la colonne cible
    target_column_name = 'Delivery_Time_min'
    assert target_column_name in data_file_df.columns, f"Test 2 ÉCHOUÉ: La colonne cible '{target_column_name}' est manquante."
    print(f"Test 2 RÉUSSI: Colonne cible '{target_column_name}' présente.")

    # Test 2: Vérifier que la colonne cible n'est pas entièrement vide
    assert not data_file_df[target_column_name].isnull().all(), "Test 3 ÉCHOUÉ: La colonne cible est entièrement vide."
    print("Test 3 RÉUSSI: Colonne cible contient des valeurs.")

    print("Tous les tests unitaires initiaux ont RÉUSSI.")


# --- Exécution principale du script

try:
    # --- Chargement des données ---
    Data_File = loading_data()

    # --- Lancer les tests unitaires initiaux ---
    run_unit_tests(Data_File)

    # --- Définition des caractéristiques (X) et de la cible (Y) ---
    TARGET_COLUMN = 'Delivery_Time_min'

    # Exclure la cible et toute autre colonne non pertinente (ex: 'ID_Commande' si elle existe)
    # Pour un projet débutant, nous supposons 'customerID' est une colonne à ignorer.
    COLUMNS_TO_DROP_FROM_X = [TARGET_COLUMN, 'customerID']  # Ajoutez d'autres ID si besoin

    X = Data_File.drop(columns=COLUMNS_TO_DROP_FROM_X,
                       errors='ignore')  # 'errors=ignore' pour ne pas échouer si une colonne n'existe pas
    Y = Data_File[TARGET_COLUMN]

    # --- Vérification de la cible pour la classification ---
    # Si Delivery_Time_min est un temps continu (ex: 25.5), c'est de la REGRESSION.
    # Pour cet exemple de CLASSIFICATION, nous allons supposer que Y a des valeurs discrètes ou est binarisée.
    # Si Y est continue, vous devrez changer vers des modèles et métriques de régression.

    # Identifier les colonnes numériques et catégorielles pour le ColumnTransformer
    numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(include='object').columns.tolist()

    print(f"\nCaractéristiques numériques détectées : {numerical_cols}")
    print(f"Caractéristiques catégorielles détectées : {categorical_cols}")

    # --- Split des données d'entraînement et de test ---
    # Utilisation de stratify=Y est crucial pour la classification avec déséquilibre de classes.
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

    print("\n--- Après le Split des Données ---")
    print(f'Taille de X_train : {X_train.shape}')
    print(f'Taille de X_test : {X_test.shape}')
    print(f'Taille de Y_train : {y_train.shape}')
    print(f'Taille de Y_test : {y_test.shape}')
    print(f"Distribution des classes dans y_train :\n {y_train.value_counts(normalize=True)}")
    print(f"Distribution des classes dans y_test :\n {y_test.value_counts(normalize=True)}")

    # --- Création du Préprocesseur (ColumnTransformer) ---
    # Pipeline pour les caractéristiques numériques
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),  # Imputation par la moyenne
        ('scaler', StandardScaler())  # Mise à l'échelle
    ])

    # Pipeline pour les caractéristiques catégorielles
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Imputation par le mode
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  # Encodage One-Hot
    ])

    # Combine les transformateurs pour différentes colonnes
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='passthrough'  # Conserve les colonnes non spécifiées (si ID n'a pas été retiré)
    )

    # --- Construction du Pipeline complet avec un classifieur de base ---
    # Ce pipeline sera la "base" sur laquelle Grid Search va travailler
    full_pipeline_base = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, solver='liblinear'))  # Classifieur par défaut
    ])

    # --- Définition de la grille d'hyperparamètres pour Grid Search ---
    # Nous explorons deux types de classifieurs et leurs hyperparamètres,
    # ainsi que la stratégie d'imputation numérique.
    param_grid = [
        {
            'classifier': [LogisticRegression(random_state=42, solver='liblinear')],  # Algorithme 1
            'classifier__C': [0.1, 1.0, 10.0],  # Hyperparamètres spécifiques à LogisticRegression
            'preprocessor__num__imputer__strategy': ['mean', 'median']
            # Hyperparamètres de l'étape d'imputation numérique
        },
        {
            'classifier': [RandomForestClassifier(random_state=42)],  # Algorithme 2
            'classifier__n_estimators': [50, 100],
            # Hyperparamètres spécifiques à RandomForest (réduit pour la simplicité)
            'classifier__max_depth': [None, 10],  # (réduit pour la simplicité)
            'preprocessor__num__imputer__strategy': ['mean', 'median']
            # Hyperparamètres de l'étape d'imputation numérique
        }
    ]

    print("\n--- Lancement de Grid Search ---")
    # make_scorer(accuracy_score) est utilisé car Grid Search est un optimisateur
    # et a besoin d'une métrique "scorée".
    grid_search = GridSearchCV(
        full_pipeline_base, 
        param_grid, 
        cv=3, 
        scoring=make_scorer(accuracy_score), 
        verbose=1,
        n_jobs=-1)

    # Entraînement de Grid Search (cela entraîne plusieurs pipelines)
    grid_search.fit(X_train, y_train)

    print("\n--- Résultats de Grid Search ---")
    print(f"Meilleure combinaison d'hyperparamètres trouvée : {grid_search.best_params_}")
    print(f"Meilleur score de validation croisée : {grid_search.best_score_:.4f}")

    # Le meilleur modèle trouvé par Grid Search est automatiquement stocké dans .best_estimator_
    best_model_pipeline = grid_search.best_estimator_

    # --- Évaluation du MEILLEUR modèle sur l'ensemble de test final ---
    print("\n--- Évaluation du MEILLEUR modèle sur l'ensemble de test ---")
    final_predictions = best_model_pipeline.predict(X_test)
    final_accuracy = accuracy_score(y_test, final_predictions)

    print(f"Précision (Accuracy) finale du MEILLEUR modèle : {final_accuracy:.4f}")
    print("\nRapport de classification final :\n", classification_report(y_test, final_predictions))

    print("\nScript terminé avec succès.")

except FileNotFoundError as e:
    print(f"ERREUR : {e}")
    print("Veuillez vérifier le chemin de votre fichier CSV et la structure des dossiers.")
except KeyError as e:
    print(f"ERREUR : Colonne manquante. {e}")
    print("Veuillez vérifier les noms des colonnes dans votre DataFrame (cible, ID, etc.).")
except Exception as e:
    print(f"Une erreur inattendue est survenue : {e}")
