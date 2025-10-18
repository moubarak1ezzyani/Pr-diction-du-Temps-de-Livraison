import transformers
from sklearn import pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

#loading data
def loading_data():
    Current_Path = os.getcwd()
    csv_path = os.path.join(Current_Path, "..", "Data", "DataSetFile_Livraison.csv")
    Data_File_func = pd.read_csv(csv_path)
    return Data_File_func

#def OneHotEncoding(Data_File,Data_File_Obj):
    Data_File_Obj = Data_File.select_dtypes(['object'])
    encoder = OneHotEncoder(sparse_output=False)
    encoder_data = encoder.fit_transform(Data_File_Obj)
    return encoder_data

#def StandardScaling(Data_File,Data_File_Obj):
    Data_File_Num = Data_File.select_dtypes(['object'])
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(Data_File_Num)
    return scaled_data

Data_File=loading_data()

X=Data_File.drop(columns=['Delivery_Time_min'])
Y=Data_File['Delivery_Time_min']

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2,random_state=42)
#Etapes:
# - Imputer : Missed Value <=> Eplucher
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
    ("classifier", LogisticRegression())
    #("classifier", RandomForestClassifier())
])
# Entrainement en une seule fois
full_pipeline.fit(X_train, y_train)
predictions=full_pipeline.predict(X_test)


print(f"Score r2 du pipeline : {r2_score(y_test, predictions)}")
