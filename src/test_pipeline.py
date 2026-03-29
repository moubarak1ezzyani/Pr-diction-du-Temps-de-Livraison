import transformers
import pytest
import os
from sklearn import pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_iris


#loading data
def loading_data():
    Current_Path = os.getcwd()
    csv_path = os.path.join(Current_Path, "..", "Data", "DataSetFile_Livraison.csv")
    Data_File_func = pd.read_csv(csv_path)
    return Data_File_func

"""def OneHotEncoding(Data_File,Data_File_Obj):
    Data_File_Obj = Data_File.select_dtypes(['object'])
    encoder = OneHotEncoder(sparse_output=False)
    encoder_data = encoder.fit_transform(Data_File_Obj)
    return encoder_data 

#def StandardScaling(Data_File,Data_File_Obj):
    Data_File_Num = Data_File.select_dtypes(['object'])
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(Data_File_Num)
    return scaled_data"""

Data_File=loading_data()
Column_X_Cleaned = Data_File.drop(columns=['Delivery_Time_min','Time_of_Day','Vehicle_Type'])
X=Data_File.drop(columns=Column_X_Cleaned,  errors='ignore')  # 'errors=ignore' pour ne pas échouer si une colonne n'existe pas
Y=Data_File['Delivery_Time_min']

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2,random_state=42)

#Etapes:
# - Imputer : Missed Value <=> Eplucher les donnees
# - Scaler : Mise à l'echelle <=> couper uniformement
# - Model : Model de Classification <=> mettre au four

#steps
Imputing_Missed=Pipeline([
    ("imputer", SimpleImputer(strategy="name"))


])
Scaling_num=Pipeline([
    ("Sacaling", StandardScaler())
])
Encondig_Obj=Pipeline([
    ("Enconding", OneHotEncoder())
])

Data_File_Num = Data_File.select_dtypes(np.number)
Data_File_Obj = Data_File.select_dtypes(['object'])

preprocessor = ColumnTransformer(
    transformers[
    ('numeric', Scaling_num, Data_File_Num),
    ('categorical', Encondig_Obj, Data_File_Obj),
])

full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ("classifier - Model", LogisticRegression())
    #("classifier", RandomForestClassifier())
    #("classifier", SVC())
])
# Entrainement en une seule fois
full_pipeline.fit(X_train, y_train)
predictions=full_pipeline.predict(X_test)


print(f"Score r2 du pipeline : {r2_score(y_test, predictions)}")

# -----Implementation de GridSearchCV
# --- Grille de Recherche
var_grid_SVC = {
    'model__C': [0.1, 1, 10],           # "__" est compté, et specifie un modèle en scikit-learn
    'model__kernel': ['linear', 'rbf']
}

var_grid_RandF = {
    'model__n_estimators': [50, 100, 150],       # Nombre d'arbres : Le nombre de membres dans votre comité d'experts.
    'model__max_depth': [3, 5, 10, None],       # Profnd max des arbres : Le nombre de questions qu'un expert a le droit de poser avant de donner un avis
    'model__min_samples_split': [2, 5]          # => Nb min d'échantillons pour diviser un nœud : Une règle de gestion du comité. "Ne créez pas un sous-comité
                                                # (une nouvelle branche) s'il y a moins de X personnes à interroger."""
}

var_grid_LogReg = {
    'model__C': [0.1, 1.0, 10],            # Tester forces de régularisation : Exactement la même que pour SVC ! C'est l'inverse de la force de régularisation.
    'model__penalty': ['l1', 'l2'],        # Tester les pénalités L1 et L2 : réduire le budget (régulariser).
    'model__solver': ['liblinear']         # 'liblinear' est un bon solver qui gère L1 et L2 : trouver la meilleure frontière.
}

#model = SVC()
grid_search_Main = GridSearchCV(
    estimator=full_pipeline,  # <--- On passe le pipeline ici. Sans pipeline : on affecte model
    param_grid = var_grid_LogReg, 
    cv = 5, 
    verbose = 1,          # 1 signifie qu'il affichera la progr et infoS générales. Valeurs comme (2, 3) affichent plus de détails.
    n_jobs = 2
)


grid_search.fit(X, y) # Entrainement sur 4/5 dans ce cas (sans un de test)

# --- 6. Afficher les résultats ---
print("\n--- Résultats de la recherche (avec Pipeline) ---")

print(f"Meilleur score (précision) : {grid_search_Main.best_score_:.4f}")       # .score() sur le 1/5 transformé : test

# Les meilleurs paramètres auront le préfixe 'model__'
print(f"Meilleurs hyperparamètres : {grid_search_Main.best_params_}")       # Soit en pipeline ou non, cette ligne ne change pas

# --- 7. Utiliser le meilleur pipeline ---
# .best_estimator_ est maintenant un Pipeline complet, 
# entraîné avec les meilleurs hyperparamètres sur l'ensemble des données.
best_pipeline = grid_search.best_estimator_

print(f"\nLe meilleur pipeline est prêt : \n{best_pipeline}")

# Prédiction avec le pipeline (il gère le scaling + la prédiction)
prediction = best_pipeline.predict([
    [5.1, 3.5, 1.4, 0.2]            # Un exemple de fleur Iris
])
print(f"Prédiction : {iris.target_names[prediction]}")      # On peut éliminer iris si vous n'avez besoin QUE de la prédiction numérique brute