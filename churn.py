# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 22:39:29 2025

@author: dell
"""

# Manipulation de données 
import pandas as pd 
import numpy as np 

# Visualisation 
import matplotlib.pyplot as plt 
import seaborn as sns 

# Division des données 
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import GridSearchCV 

# Prétraitement 
from sklearn.preprocessing import StandardScaler 
from sklearn.impute import SimpleImputer 
from sklearn.pipeline import Pipeline 

# Algorithme K-NN 
from sklearn.neighbors import KNeighborsClassifier 

# Métriques de classification 
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import roc_curve 
from sklearn.metrics import accuracy_score

import joblib

# --- 1. Exploration et Visualisation
#Chargement des données
df = pd.read_csv("churn_clients.csv")
colonnes = ['customer_id','satisfaction','nb_reclamations','temps_reponse_support','utilisation_plateforme','montant_dernier_achat','nb_produits_achetes','a_utilise_promo','churn']

print("Données chargées :", df.shape)

print("\nAperçu des données:") 
print(df.head())

print("\nStatistiques descriptives:") 
print(df.describe())

#Valeurs manquantes de client_ecommerce
print("\nValeurs manquantes :")
print(df.isna().sum())

#Supprime les valeurs manquantes : 
df = df.dropna()

# Analyse de la variable cible
print("\nDistribution de la variable cible (churn):")
print(df["churn"].value_counts(normalize=True))

# Distribution de la variable cible 
print("\n Distribution de la variable cible:") 
target_dist = df['churn'].value_counts() 
print(target_dist) 

print(f"\nPourcentage de clients restés: {(1-df['churn'].mean())*100:.2f}%") 
plt.figure(figsize=(10, 6)) 
plt.subplot(1, 2, 1) 
sns.countplot(data=df, x='churn', palette=['lightblue', 'salmon']) 

plt.title("Répartition des clients") 
plt.xlabel("Présence de la clientèle(0=Resté, 1=Parti)") 
plt.ylabel("Nombre de clients") 
plt.subplot(1, 2, 2) 
plt.pie(target_dist.values, labels=['Parti', 'Resté'], autopct='%1.1f%%', colors=['salmon', 'lightblue'], startangle=180) 

plt.title("Distribution de la clientèle") 
plt.tight_layout() 
plt.show()

print("\n=== PRÉPARATION DES DONNÉES ===")
# Séparation des caractéristiques et de la variable cible 
X = df.drop("churn", axis=1) 
y = df["churn"] 

# Division en ensembles d'entraînement et de test (80% / 20%) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


#Préparation des données
# Séparation des features(paramèrtres) et de la cible 
X = df.drop(columns=["churn", "customer_id"]) 
y = df["churn"] 
preprocessing_pipeline = Pipeline([ 
    ("imputer", SimpleImputer(strategy="mean")), 
    ("scaler", StandardScaler()) 
])

# Application du pipeline 
X_preprocessed = preprocessing_pipeline.fit_transform(X)

#Division en ensembles d'entraînement et de test (80/20) avec stratification
X_train, X_test, y_train, y_test = train_test_split( 
    X_preprocessed, y, test_size=0.2, random_state=42, stratify=y 
)

print(f"\nDimensions de X_train: {X_train.shape}") 
print(f"\nDimensions de X_test: {X_test.shape}")


#Construction du modèle K-NN
knn_model = KNeighborsClassifier(n_neighbors=5) 

# Entraînement du modèle 
knn_model.fit(X_train, y_train) 

# Prédictions 
y_pred = knn_model.predict(X_test) 
y_proba = knn_model.predict_proba(X_test)[:, 1]

# --- Évaluation du modèle K-NN ---
# Visualisation de la matrice de confusion 
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nMatrice de confusion :")
print(conf_matrix)

# Visualisation de la matrice de confusion
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Resté (0)', 'Parti (1)'],
            yticklabels=['Resté (0)', 'Parti (1)'])
plt.xlabel('Prédiction')
plt.ylabel('Client')
plt.title('Matrice de confusion')
plt.show()

# Rapport de classification 
print("\nRapport de classification :") 
print(classification_report(y_test, y_pred))

# Courbe ROC 
roc_auc = roc_auc_score(y_test, y_proba) 
fpr, tpr, thresholds = roc_curve(y_test, y_proba) 
plt.figure(figsize=(8, 6)) 
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}") 
plt.plot([0, 1], [0, 1], "k--") 
plt.title("Courbe ROC") 
plt.xlabel("Taux de faux positifs") 
plt.ylabel("Taux de vrais positifs") 
plt.legend() 
plt.grid() 
plt.show()

#Optimisation du modèle K-NN
param_grid = {'n_neighbors': list(range(1, 31))}  # Tester k de 1 à 30 
grid_search = GridSearchCV( 
KNeighborsClassifier(),  
param_grid,  
cv=5,             

# Validation croisée en 5 plis 
scoring='roc_auc',  # Optimiser l'AUC 
verbose=1 
) 
grid_search.fit(X_train, y_train) 

# Afficher les résultats 
print(f"Meilleure valeur de k: {grid_search.best_params_['n_neighbors']}") 
print(f"Meilleur score (AUC): {grid_search.best_score_:.4f}") 

# Visualiser les résultats de la recherche 
plt.figure(figsize=(10, 6)) 
cv_results = pd.DataFrame(grid_search.cv_results_) 
plt.errorbar( 
cv_results['param_n_neighbors'],  
cv_results['mean_test_score'],  
yerr=cv_results['std_test_score'] 
) 
plt.title('AUC par valeur de k') 
plt.xlabel('Nombre de voisins (k)') 
plt.ylabel('Score AUC') 
plt.grid(True) 
plt.show() 

# Utiliser le meilleur modèle 
best_knn = grid_search.best_estimator_ 
y_pred_best = best_knn.predict(X_test) 
y_proba_best = best_knn.predict_proba(X_test)[:, 1] 

# Évaluation du meilleur modèle 
print("\nÉvaluation du modèle optimisé:") 
print(classification_report(y_test, y_pred_best)) 

# Courbe ROC du modèle optimisé 
roc_auc_best = roc_auc_score(y_test, y_proba_best) 
fpr_best, tpr_best, thresholds_best = roc_curve(y_test, y_proba_best) 
plt.figure(figsize=(8, 6)) 
plt.plot(fpr, tpr, 'b--', label=f"K=5, AUC = {roc_auc:.2f}") 
plt.plot(fpr_best, tpr_best, 'r-', label=f"K={grid_search.best_params_['n_neighbors']}, AUC = {roc_auc_best:.2f}") 
plt.plot([0, 1], [0, 1], "k--") 
plt.title("Comparaison des courbes ROC") 
plt.xlabel("Taux de faux positifs") 
plt.ylabel("Taux de vrais positifs") 
plt.legend() 
plt.grid() 
plt.show()

