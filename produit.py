# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 15:15:29 2025

@author: dell
"""

# Manipulation de données 
import pandas as pd 
import numpy as np 

# Visualisation 
import matplotlib.pyplot as plt 
import seaborn as sns 

# Division et prétraitement 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 

# Métriques 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import roc_curve 
from sklearn.metrics import roc_auc_score 

# TensorFlow/Keras - Deep Learning 
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.models import load_model 
from tensorflow.keras.layers import Dense 
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization 
from tensorflow.keras.callbacks import EarlyStopping 
from tensorflow.keras.callbacks import ModelCheckpoint 
from tensorflow.keras.optimizers import Adam 

# Sauvegarde de modèles 
import joblib

# --- 1. Préparation et Analyse Exploratoire 
print("=== ANALYSE EXPLORATOIRE ET PRÉPARATION ===")

# Chargement des données
df = pd.read_csv("produits_clients.csv")
print("Données chargées :", df.shape)

print("\nStatistiques descriptives:") 
print(df.describe())

# Aperçu
print("\nAperçu des données : ")
print(df.head())
print(df.info())

# Vérification des valeurs manquantes
print("\nValeurs manquantes :")
print(df.isna().sum())
# Suppression des lignes avec valeurs manquantes si peu nombreuses, sinon imputation
df = df.dropna()

# Visualisation
plt.figure(figsize=(8,8))
df['categorie_preferee'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title("Distribution des catégories de produits")
plt.show()

plt.figure(figsize=(10,6))
sns.countplot(data=df, x='categorie_preferee', hue='sexe')
plt.title("Préférences par sexe")
plt.show()

plt.figure(figsize=(10,6))
sns.boxplot(data=df, x='categorie_preferee', y='age')
plt.title("Préférences par âge")
plt.show()

# --- Préparation des données ---

# On exclut customer_id qui n'est pas une feature prédictive généralisable
X = df.drop(['customer_id', 'produit_suivant'], axis=1)
y = df['produit_suivant']

# Encodage des variables catégorielles
X = pd.get_dummies(X, columns=['categorie_preferee'], prefix='cat_pref')


# Division Train/Test 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Sauvegarde du scaler pour la fonction de recommandation
joblib.dump(scaler, 'scaler.pkl')

print("X_train shape:", X_train_scaled.shape)
print("y_train shape:", y_train.shape)

# --- 2. Construction d’un Réseau de Neurones (Modèle 1) ---
print("\n=== CONSTRUCTION DU MODÈLE 1 ===")

nombre_classes = y.nunique()
input_dim = X_train_scaled.shape[1]

model1 = Sequential([
    Dense(64, activation='relu', input_shape=(input_dim,)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(nombre_classes, activation='softmax')
])

model1.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])

model1.summary()

# Entraînement
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history1 = model1.fit(X_train_scaled, y_train,
                    validation_split=0.2,
                    epochs=50,
                    batch_size=32,
                    callbacks=[early_stopping],
                    verbose=1)

# Courbes Loss et Accuracy
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history1.history['loss'], label='Train Loss')
plt.plot(history1.history['val_loss'], label='Val Loss')
plt.title('Model 1 Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history1.history['accuracy'], label='Train Acc')
plt.plot(history1.history['val_accuracy'], label='Val Acc')
plt.title('Model 1 Accuracy')
plt.legend()
plt.savefig('model1_curves.png') # Sauvegarde au lieu d'afficher
# plt.show()

# --- 3. Évaluation et Amélioration ---
print("\n=== ÉVALUATION MODÈLE 1 ===")
y_pred1 = np.argmax(model1.predict(X_test_scaled), axis=1)
print("Accuracy:", accuracy_score(y_test, y_pred1))
print("\nRapport de classification :\n", classification_report(y_test, y_pred1))

cm1 = confusion_matrix(y_test, y_pred1)
plt.figure(figsize=(8,6))
sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(y_test),
            yticklabels=np.unique(y_test))
plt.title("Matrice de confusion - Modèle 1")
plt.xlabel("Prédictions")
plt.ylabel("Valeurs réelles")
plt.show()

# --- Modèle 2 (Architecture Améliorée) ---
print("\n=== CONSTRUCTION DU MODÈLE 2 (AMÉLIORÉ) ===")
# Plus profond, plus de neurones, BatchNormalization
model2 = Sequential([
    Dense(256, activation='relu', input_shape=(input_dim,)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(nombre_classes, activation='softmax')
])

model2.compile(optimizer=Adam(learning_rate=0.001),
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])

history2 = model2.fit(X_train_scaled, y_train,
                    validation_split=0.2,
                    epochs=100, # Plus d'epochs potentiellement
                    batch_size=32,
                    callbacks=[early_stopping],
                    verbose=1)

# --- Courbes Loss et Accuracy - Modèle 2 ---
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history2.history['loss'], label='Train Loss')
plt.plot(history2.history['val_accuracy'], label='Validation Loss')
plt.title('Évolution de la Loss - Modèle 2')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history2.history['accuracy'], label='Train Acc')
plt.plot(history2.history['val_loss'], label='Validation Accuracy')
plt.title('Évolution de l\'Accuracy'' - Modèle 2')
plt.legend()


print("\n=== ÉVALUATION MODÈLE 2 ===")
y_pred2 = np.argmax(model2.predict(X_test_scaled), axis=1)
print("Accuracy:", accuracy_score(y_test, y_pred2))
print("\nRapport de classification :\n", classification_report(y_test, y_pred2))

cm2 = confusion_matrix(y_test, y_pred2)
plt.figure(figsize=(8,6))
sns.heatmap(cm2, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(y_test),
            yticklabels=np.unique(y_test))
plt.title("Matrice de confusion - Modèle 2")
plt.xlabel("Prédictions")
plt.ylabel("Valeurs réelles")
plt.show()


# Sauvegarde du meilleur modèle
model2.save('best_model.h5')
print("Meilleur modèle sauvegardé sous 'best_model.h5'")


# --- Fonction de Recommandation ---
def recommander_produit(customer_id, age, sexe, historique, categorie_preferee_input):
    """
    Recommande une catégorie de produit pour un client donné.
    Attention: Il faut fournir 'categorie_preferee' car elle fait partie des features d'entraînement.
    """
    # Création du DataFrame pour l'input
    # Il faut reconstruire la structure exacte des features (X)
    
    # 1. Créer un dictionnaire de base
    data = {
        'age': [age],
        'sexe': [sexe]
    }
    # Ajouter l'historique (hist_0 à hist_9)
    for i, val in enumerate(historique):
        data[f'hist_{i}'] = [val]
        
    # Créer le DF
    input_df = pd.DataFrame(data)
    
    # 2. Gérer le One-Hot Encoding de categorie_preferee
    # On doit avoir les colonnes cat_pref_0, cat_pref_1, etc.
    # On va créer ces colonnes à 0 et mettre à 1 celle qui correspond
    # Pour savoir quelles colonnes existent, on peut regarder X.columns (si on l'avait gardé accessible)
    # Ou on sait qu'il y a 5 catégories (0,1,2,3,4) d'après l'EDA
    
    for i in range(5): # Supposons 5 catégories 0-4
        col_name = f'cat_pref_{i}'
        if i == categorie_preferee_input:
            input_df[col_name] = 1
        else:
            input_df[col_name] = 0
            
    # S'assurer que l'ordre des colonnes est le même que lors du fit du scaler
    # Le scaler a été fit sur X_train.columns
    # On doit réordonner input_df selon les colonnes de X (qui sont stockées implicitement dans le scaler si on utilise sklearn < 1.0, mais mieux vaut être explicite)
    # Ici on va réutiliser la liste des colonnes de X
    input_df = input_df[X.columns]
    
    # 3. Normalisation
    input_scaled = scaler.transform(input_df)
    
    # 4. Prédiction
    prediction_proba = model2.predict(input_scaled, verbose=0)
    categorie_predite = np.argmax(prediction_proba)
    
    return categorie_predite, prediction_proba

print("\n=== TEST DE LA FONCTION DE RECOMMANDATION ===")
# Test sur 3 profils
# Profil 1: Jeune, Homme, Préfère cat 0, Historique varié
hist1 = [0.1]*10
pred1, prob1 = recommander_produit(999, 25, 1, hist1, 0)
print(f"Client 1 (25 ans, H, Pref 0) -> Recommandation : {pred1}")

# Profil 2: Senior, Femme, Préfère cat 3, Historique ciblé
hist2 = [0.9 if i==3 else 0.1 for i in range(10)]
pred2, prob2 = recommander_produit(888, 60, 0, hist2, 3)
print(f"Client 2 (60 ans, F, Pref 3) -> Recommandation : {pred2}")

# Profil 3: Moyen âge, Homme, Préfère cat 1
hist3 = [0.5]*10
pred3, prob3 = recommander_produit(777, 40, 1, hist3, 1)
print(f"Client 3 (40 ans, H, Pref 1) -> Recommandation : {pred3}")


