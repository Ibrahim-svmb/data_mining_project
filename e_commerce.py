# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 13:27:20 2025

@author: dell
"""

# Manipulation de données 
import pandas as pd 
import numpy as np 

# Visualisation 
import matplotlib.pyplot as plt 
import seaborn as sns 

# Prétraitement 
from sklearn.preprocessing import StandardScaler 

# Algorithmes de clustering 
from sklearn.cluster import KMeans 
from sklearn.cluster import DBSCAN 
from sklearn.cluster import AgglomerativeClustering 

# Métriques d'évaluation 
from sklearn.metrics import silhouette_score 
from sklearn.metrics import davies_bouldin_score 
from sklearn.metrics import calinski_harabasz_score

# Réduction de dimensionnalité 
from sklearn.decomposition import PCA 

# Visualisation 3D 
from mpl_toolkits.mplot3d import Axes3D 

# Analyse statistique 
from sklearn.feature_selection import f_classif 

# --- 1. Exploration et Visualisation
#Chargement des données
df = pd.read_csv("clients_ecommerce.csv")
colonnes = ['customer_id','age','revenu_annuel','score_depense','frequence_achat','panier_moyen','anciennete'] 

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

#Matrice de corrélation
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matrice de corrélation - clients_ecommerce")
plt.show()

#Visualisation de distribution de trois variables pertinentes
plt.figure(figsize=(20,12))
for i, col in enumerate(["revenu_annuel", "score_depense", "frequence_achat"], 1):
    plt.subplot(2,2,i)
    sns.histplot(df[col], kde=True, bins=30, color="skyblue")
    plt.title(f"Distribution de {col}")
plt.tight_layout()
plt.show()


# --- 2. Clustering K-Means
features = df[['age','revenu_annuel','score_depense','frequence_achat','panier_moyen','anciennete']] 

# Normalisation des données 
scaler = StandardScaler() 
scaled_features = scaler.fit_transform(features)

# Boucle sur k = 3, 4, 5
for k in [3,4,5]:
    print(f"\n===== K-Means avec k={k} =====")
    
    # Application de K-Means
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    df[f"Cluster_KMeans_{k}"] = kmeans.fit_predict(scaled_features)
    
    # Visualisation 2D
    plt.figure(figsize=(8,6))
    sns.scatterplot(
        x=df["revenu_annuel"],
        y=df["panier_moyen"],
        hue=df[f"Cluster_KMeans_{k}"],
        palette="viridis",
        s=50
    )
    plt.title(f"K-Means Clustering via PCA (k={k})")
    plt.xlabel("Revenu annuel")
    plt.ylabel("Panier moyen")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Visualisation 3D
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        df["revenu_annuel"],
        df["panier_moyen"],
        df["frequence_achat"],
        c=df[f"Cluster_KMeans_{k}"],
        cmap="viridis",
        s=50,
        alpha=0.6
    )
    ax.set_xlabel("Revenu annuel")
    ax.set_ylabel("Panier moyen")
    ax.set_zlabel("Fréquence d'achat")
    ax.set_title(f"Clustering K-Means en 3D (k={k})")
    legend = ax.legend(*scatter.legend_elements(), title="Cluster")
    ax.add_artist(legend)
    plt.tight_layout()
    plt.show()
    
    # Caractérisation des clusters
    cluster_characteristics = df.groupby(f"Cluster_KMeans_{k}")[features.columns].mean().round(2)
    cluster_sizes = df[f"Cluster_KMeans_{k}"].value_counts().sort_index()
    
    print(f"\nCaractéristiques moyennes cluster k={k}:")
    for i in range(len(cluster_characteristics)):
        print(f"\nCluster {i} (taille : {cluster_sizes[i]} observations) :")
        print(cluster_characteristics.iloc[i])
    
    # Heatmap des moyennes
    plt.figure(figsize=(8,5))
    sns.heatmap(cluster_characteristics, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title(f"Moyennes des variables par cluster (k={k})")
    plt.tight_layout()
    plt.show()
    
    # Évaluation des performances
    silhouette = silhouette_score(scaled_features, df[f"Cluster_KMeans_{k}"])
    davies_bouldin = davies_bouldin_score(scaled_features, df[f"Cluster_KMeans_{k}"])
    calinski_harabasz = calinski_harabasz_score(scaled_features, df[f"Cluster_KMeans_{k}"])
    inertia = kmeans.inertia_
    
    print("\nMétriques d'évaluation :")
    print(f"Score de silhouette: {silhouette:.3f}")
    print(f"Indice de Davies-Bouldin: {davies_bouldin:.3f}")
    print(f"Indice de Calinski-Harabasz: {calinski_harabasz:.3f}")
    print(f"Inertie: {inertia:.3f}")
    
    # Visualisation des métriques
    plt.figure(figsize=(14,4))
    
    # Inertie
    plt.subplot(1,4,1)
    plt.bar([f"k={k}"], [inertia], color="purple")
    plt.title("Inertie\n(plus bas est meilleur)")
    plt.text(0, inertia, f"{inertia:.1f}", ha="center", va="bottom")
    
    #Silhouette
    plt.subplot(1,4,2)
    plt.bar([f"k={k}"], [silhouette], color="blue")
    plt.title("Silhouette\n(plus élevé est meilleur)")
    plt.text(0, silhouette, f"{silhouette:.3f}", ha="center", va="bottom")
    
    #Davies_bouldin
    plt.subplot(1,4,3)
    plt.bar([f"k={k}"], [davies_bouldin], color="green")
    plt.title("Davies-Bouldin\n(plus bas est meilleur)")
    plt.text(0, davies_bouldin, f"{davies_bouldin:.3f}", ha="center", va="bottom")
    
    plt.tight_layout()
    plt.show()
    
# Listes pour stocker les métriques
inertias = []
silhouettes = []

# Plage de k testés
K = range(2, 10)

for k in K:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(scaled_features)
    
    # Inertie
    inertias.append(km.inertia_)
    
    # Score de silhouette
    silhouettes.append(silhouette_score(scaled_features, km.labels_))

# Visualisation des résultats
plt.figure(figsize=(10, 4))

# Méthode du coude (Inertie)
plt.subplot(1, 2, 1)
plt.plot(K, inertias, 'bo-')
plt.xlabel('Nombre de clusters (k)')
plt.ylabel('Inertie')
plt.title('Méthode du coude')

# Score silhouette
plt.subplot(1, 2, 2)
plt.plot(K, silhouettes, 'ro-')
plt.xlabel('Nombre de clusters (k)')
plt.ylabel('Score de silhouette')
plt.title('Score silhouette en fonction de k')

plt.tight_layout()
plt.savefig('selection_k_optimal.png')
plt.show()