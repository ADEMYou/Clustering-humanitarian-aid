#!/usr/bin/env python
# coding: utf-8

# # Projet : clustering des différents pays pour guider les choix d'affectation d'aide humanitaire

# ## Étude préalable :  comparaison de différentes méthodes de clustering 

# ### Importation des modules pour tout le projet

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from scipy.cluster import hierarchy
from sklearn.ensemble import IsolationForest


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.metrics import adjusted_rand_score, silhouette_score, make_scorer, davies_bouldin_score


# ### Chargement des jeux de données pour tester les différents modèles de clustering

# In[3]:


#Chargement des données
data_jain = pd.read_csv("jain.txt", sep = '\t', header = None)
data_aggregation = pd.read_csv("Aggregation.txt", sep = "\t", header = None )
data_pathbased = pd.read_csv("pathbased.txt", sep = "\t", header = None)
data_g20 = pd.read_csv("g2-2-20.txt", sep = "\s+", header = None, )
data_g100 = pd.read_csv("g2-2-100.txt", sep= "\s+", header = None)

#Dictionnaire avec association nom du dataset - dataset (facilitant l'écriture du code dans la partie "classification des données")
datasets = {'data_jain': data_jain , 'data_aggregation': data_aggregation, 'data_pathbased': data_pathbased, 'data_g20': data_g20, 'data_g100': data_g100}


# ### Analyse des différents jeux de données
# 
# 

# In[5]:


#Infos générales sur les jeux de données
print("jain.txt", data_jain.info(), "\n")
print("Aggregation.txt", data_aggregation.info(), "\n")
print("pathbased.txt", data_pathbased.info(), "\n")
print("g2-2-20.txt", data_g20.info(), "\n")
print("g2-2-100.txt", data_g100.info())


# In[6]:


#Premières lignes des jeux de données
print("jain.txt", data_jain.head(), "\n")
print("Aggregation.txt", data_aggregation.head(), "\n")
print("pathbased.txt", data_pathbased.head(), "\n")
print("g2-2-20.txt", data_g20.head(), "\n")
print("g2-2-100.txt", data_g100.head())


# #### Visualisation des jeux de données

# In[7]:


plt.figure(figsize=(20,15))
plot_index = 1
for dataset_name, dataset in datasets.items():
    plt.subplot(2,3, plot_index)
    if dataset.shape[1] == 3:
        plt.scatter(dataset.iloc[:, 0], dataset.iloc[:, 1], c = dataset.iloc[:, 2])
    else:
        plt.scatter(dataset.iloc[:, 0], dataset.iloc[:, 1])
    plt.title(dataset_name)
    plot_index += 1
plt.show()


# #### Classification des données pour les différents modèles : KMeans, DBSCAN, classification hiérarchique ascendante et GaussianMixture

# In[28]:


scaler = StandardScaler()
#Matrice des scores de Rand ajusté (valeur 0 pour les jeux de données non étiquetés)
#en ligne les datasets (dans l'ordre de déclaration dans le dictionnaire datasets)
#en colonne les différents modèles (dans l'ordre de déclaration dans le dictionnaire models)
rand_score = np.zeros((5, 4))
#Matrice des coefficients de Silhouette (valeur 0 si le nombre de clusters est de 1)
silhouette_score_matrix = np.zeros((5, 4))

#Nombre de clusters pour chaque dataset
nb_clusters = [2, 7, 3, 2, 1]
#Paramètres du DBSCAN pour les différents datasets
dbscan_params = [[0.3, 18], [0.2, 10], [0.31, 10], [0.3, 10], [0.3, 5]]
#Paramètres pour le modèle GaussianMixture : le type de covariance
gaussian_params = ['diag', 'tied', 'spherical', 'diag', 'diag']
#Paramètres pour le modèle hiérarchique ascendant : distance entre clusters
hierarchical_params = ['weighted', 'centroid', 'ward', 'ward', 'ward']

data_index = 0
for dataset_name, dataset in datasets.items() : 
    #Conversion du dataset en tableau Numpy
    X = dataset.iloc[:, 0:2].to_numpy()
    #Standardisation
    Z = scaler.fit_transform(X)
    #Déclaration des modèles
    Kmeans = KMeans(n_clusters = nb_clusters[data_index], random_state = 0)
    Dbscan = DBSCAN(eps = dbscan_params[data_index][0], min_samples = dbscan_params[data_index][1], metric = 'euclidean')
    Gaussian = GaussianMixture(n_components = nb_clusters[data_index], covariance_type = gaussian_params[data_index], random_state = 0)
    M = hierarchy.linkage(Z, method = hierarchical_params[data_index], metric= 'euclidean')
    models = {'Kmeans': Kmeans, 'Dbscan': Dbscan, 'Gaussian_model' : Gaussian, 'Hierarchical_model': M}
    model_index = 0
    #Affichage des véritables classes pour les datasets annotés
    if dataset.shape[1] == 3:
        plt.figure(figsize=(20,8))
        plt.scatter(dataset.iloc[:, 0], dataset.iloc[:, 1], c = dataset.iloc[:, 2])
        plt.title(f'Classification réelle du dataset {dataset_name}')
    plt.figure(figsize = (20, 15))
    for name_model, model in models.items():
        #Entraînement sur les données
        if(name_model == 'Hierarchical_model'):
            result = hierarchy.fcluster(model, t = nb_clusters[data_index], criterion = 'maxclust')
        else: 
            result = model.fit_predict(Z)
        #Affichage des résultats de la classification du modèle   
        plt.subplot(2, 2, model_index + 1)
        plt.scatter(dataset.iloc[:, 0], dataset.iloc[:, 1], c=result)
        plt.title(f'Classification par {name_model} sur le dataset {dataset_name}')
        #Calcul des scores
        #Pour le calcul du coefficient de Silhouette, il doit avoir plus d'une classe
        if np.unique(result).shape[0] != 1:
            silhouette_score_matrix[data_index, model_index] = silhouette_score(Z, result)
        #Pour comparer les classes véritables et les classes trouvées par le modèle, il faut un dataset annoté
        if dataset.shape[1] == 3:
            rand_score[data_index, model_index] = adjusted_rand_score(dataset.iloc[:, 2], result)
        model_index += 1
                  
    data_index += 1


# ### Scores obtenus pour les différentes métriques 

# In[17]:


print("Adjusted Rand score \n", rand_score, "\n")
print("\n Silhouette score \n", silhouette_score_matrix)


# ## Clustering des pays en fonction de leur développement

# ### Examen des données

# #### Analyse du jeu de données

# In[361]:


data = pd.read_csv('data.csv')
print(data.shape)
print("Global information ", data.info())
print("\n Statistics \n \n", data.describe())
print("\n First lines \n \n", data.head())


# In[362]:


data


# #### Visualisation

# In[363]:


sn.pairplot(data)
plt.show()


# In[364]:


plt.figure(figsize=(20,15))
for index,feature in enumerate(data.columns[1:]): 
    plt.subplot(3,3, index+1)
    sn.distplot(data[feature])
plt.show()    


# #### Examen des valeurs manquantes

# In[365]:


data.isna().sum(axis=0)


# In[366]:


#Pays dont certaines données sont manquantes
print(data[data.isna()['total_fertility']]['country'])
print(data[data.isna()['GDP']]['country'])


# ### Traitement des données 

# #### Valeurs manquantes

# In[368]:


#Possibilité 1 : remplacer les valeurs manquantes par la moyenne avec fillna
#data.fillna(value={'total_fertility': data['total_fertility'].mean(), 'GDP': data['GDP'].mean()}, inplace=True)
#Possibilité 2 : remplacer par les véritables données trouvées sur Internet

data.loc[54, 'total_fertility'] = 2
data.loc[112, 'total_fertility'] = 7.2
data.loc[75, 'GDP'] = 35000
data.loc[114, 'GDP'] = 90000
data.info()


# #### Détection d'anomalies

# In[369]:


filter =  IsolationForest()
filter.fit(data.iloc[:, 1:])
anomaly = filter.predict(data.iloc[:, 1:])


# In[370]:


data['anomaly'] = anomaly
sn.pairplot(data, hue='anomaly')
plt.show()
print(data[data['anomaly'] == -1])


# ##### Traitement des valeurs aberrantes

# In[371]:


#Traitement des PIB pour l'Australie, les USA et le Royaume-Uni
data.loc[7, 'GDP'] = 59000
data.loc[158, 'GDP'] = 43000
data.loc[159, 'GDP'] = 56000
#Traitement de l'espérance de vie pour le Bangladesh
data.loc[12, 'life_expectation'] = 71
print(data.describe())


# #### Standardisation 

# In[372]:


X = data.iloc[:, 1:-1].to_numpy()
scaler = StandardScaler()
Z = scaler.fit_transform(X)
print(Z.shape)
print(Z.mean(axis=0))


# #### Visualisation après nettoyage et réduction des données

# In[373]:


transformed_data = pd.DataFrame(Z, columns= data.columns[1:-1])
sn.pairplot(transformed_data)
plt.show()


# In[374]:


plt.figure(figsize=(20,15))
for index,feature in enumerate(data.columns[1:-1]): 
    plt.subplot(3,3, index+1)
    sn.distplot(transformed_data[feature])
plt.show()  


# ### Corrélations

# In[377]:


transformed_data.corr()


# In[378]:


sn.heatmap(transformed_data.corr())


# In[379]:


pd.plotting.scatter_matrix(transformed_data, figsize = (25,20))
plt.show()


# ### Clustering (sans ACP)

# In[393]:


#trouver les meilleurs paramètres pour DBSCAN
def best_model_Dbscan(data, eps, min_samples):
    #Initialisation des paramètres (il doit avoir plus d'un cluster)
    i,j = 0,0
    while np.unique(DBSCAN(eps=eps[i], min_samples = min_samples[j]).fit_predict(data)).shape[0] < 2:
        i += 1
        j += 1
    best_model = DBSCAN(eps=eps[i], min_samples = min_samples[j])
    best_score = silhouette_score(data, best_model.fit_predict(data))
    #on teste toutes les combinaisons de paramètres
    for value in eps:
        for min_sample in min_samples:
            model = DBSCAN(eps=value, min_samples = min_sample)
            if np.unique(model.fit_predict(data)).shape[0] >= 2:
                score = silhouette_score(data, model.fit_predict(data))
                #on conserve celui qui a le meilleur score de Silhouette
                if score > best_score:
                   best_model = model
                   best_score = score
    return best_model
 
#va contenir les scores pour chaque valeur de K et pour chaque modèle : KMeans (col 0), GaussianMixture(col 1), 
#CAH (col 2) et DBSCAN (col 3)
K_silhouette_scores = np.zeros((19, 4))
#Modèle optimal pour DBSCAN    
eps = np.arange(0.1, 2, 0.1)
min_samples = range(3, 20)
Dbscan = best_model_Dbscan(Z, eps, min_samples)
print(Dbscan.get_params())
clusters = Dbscan.fit_predict(Z)
print("Nombre de clusters pour DBSCAN : ", clusters)
#Calcul du score de Silhouette pour DBSCAN
K_silhouette_scores[:, 3] = silhouette_score(Z, clusters) 
M = hierarchy.linkage(Z, method = 'average', metric= 'euclidean')
#Clustering pour différents nombres de classe
#Matrice pour contenir les inerties intra-classe pour chaque valeur de K et chaque modèle
inertia = []
for K in range(2,21):
    K_model_scores = []
    #Différents modèles
    models = {'KMeans': KMeans(n_clusters = K, random_state=0),'Gaussian': GaussianMixture(n_components = K, covariance_type = 'spherical', random_state=0), 'Hierarchical' : M}
    index = 0
    for model_name, model in models.items():
        if(model_name == 'Hierarchical'):
            clusters = hierarchy.fcluster(model, t = K, criterion = 'maxclust')
        else:
            clusters = model.fit_predict(Z)
        #Calcul du score de silhouette obtenu pour ce modèle
        K_silhouette_scores[K-2, index] = silhouette_score(Z, clusters)
        #Pour KMeans, on calcule l'inertie pour chaque valeur de K => utile pour déterminer K (méthode du coude)
        if model_name == 'KMeans':
           inertia.append(model.inertia_)
        index += 1  


# #### Détermination du nombre de clusters K 

# In[394]:


print("Score obtenu pour différentes valeurs de K pour chaque modèle \n", K_silhouette_scores)
print("\n Score pour chaque modèle moyenné sur les valeurs de K \n",K_silhouette_scores.mean(axis=0))
print("\n Score pour chaque valeur de K moyenné sur les différents modèles \n", K_silhouette_scores.mean(axis=1))
#On affiche l'évolution du score de Silhouette en fonction de K
plt.figure(figsize=(15,10))
plt.plot(range(2,21),  K_silhouette_scores.mean(axis=1))
plt.xlabel('K nombre de clusters')
plt.ylabel('Coefficient de Silhouette')
#Méthode du coude pour déterminer le nombre de clusters à retenir
plt.figure(figsize=(15,10))
plt.plot(range(2, 21), inertia)
plt.xlabel('K nombre de clusters')
plt.ylabel('Inertie inter-cluster')
plt.show()
#Dendrogramme
plt.figure(figsize=(15,10))
hierarchy.dendrogram(M)
plt.show()


# #### Différence des clusters entre les différents modèles 

# In[323]:


model  = KMeans(n_clusters = 3, random_state=0)
Gaussian = GaussianMixture(n_components = 5, covariance_type='spherical', random_state=0)
M = hierarchy.linkage(Z, method = 'average', metric= 'euclidean')

eps = np.arange(0.1, 2, 0.1)
min_samples = range(3, 20)
partitions = [model.fit_predict(Z), Gaussian.fit_predict(Z), hierarchy.fcluster(M, t = 5, criterion = 'maxclust'),]
rand_score_models = np.zeros((3,3))
#on calcule l'indice de Rand ajusté pour comparer les partitions obtenues entre les modèles
for i in range(3):
    for j in range(3):
        rand_score_models[i,j] = adjusted_rand_score(partitions[i], partitions[j])

print(rand_score_models)
#Intéressant de faire une méthode ensembliste (boosting par exemple)


# #### Analyse des clusters obtenus avec K = 17

# In[517]:


CAH_model = hierarchy.linkage(Z, method = 'average', metric= 'euclidean')
clusters = hierarchy.fcluster(M, t = 17, criterion = 'maxclust')
data['clusters'] = clusters
print(data.describe())
#statistiques par cluster
print(data.groupby('clusters').mean())


# #### Visualisation des nuages de points avec la partition obtenue

# In[521]:


plt.figure(figsize=(20,15))
transformed_data['clusters'] = clusters
sn.pairplot(transformed_data, hue = 'clusters')
plt.show()


# ### Clustering avec ACP normée

# In[522]:


reduction_model = PCA(n_components = Z.shape[1], copy = False)
Z = reduction_model.fit_transform(Z)


# #### Nombre d'axes à conserver

# In[525]:


#Pourcentage d'inertie expliquée
explained_inertia = np.cumsum(reduction_model.explained_variance_ratio_)
print(explained_inertia)
#Valeurs propres (règle de Kaiser)
print(reduction_model.explained_variance_)
#Eboulis des valeurs propres
plt.plot(range(1, Z.shape[1]+1), reduction_model.explained_variance_)
plt.title('Eboulis des valeurs propres')
plt.xlabel('Numéro de la valeur propre')
plt.ylabel('Valeur propre')


# #### Interprétation des axes

# In[527]:


#Nuage des individus dans le plan principal
plt.scatter(Z[:,0], Z[:,1])
plt.title('Nuage des individus projeté sur le plan principal 1-2')
plt.xlabel('Composante principale 1')
plt.ylabel('Composante principale 2')


# In[576]:


#Nuage des variables dans le plan factoriel principal
pf = np.ones((Z.shape[1], 4))
#Calcul des coordonnées des variables sur les 4 premiers facteurs principaux
for i in range(4):
   pf[:,i] = np.sqrt(reduction_model.explained_variance_[i])*reduction_model.components_[i]
plt.figure(figsize=(12,8))
plt.scatter(pf[:,0], pf[:,1])
plt.xlabel("Facteur principal 1")
plt.ylabel("Facteur principal 2")
plt.title("Représentation des variables dans le premier plan factoriel")
plt.xticks(np.arange(-1, 2, 1))
plt.yticks(np.arange(-1, 2, 1))
for i in range(pf.shape[0]):
    plt.text(pf[i,0], pf[i,1], s=transformed_data.columns[i])


# #### Clustering

# In[601]:


Z = scaler.fit_transform(X)
reduction_model = PCA(n_components = 5, copy = False)
Z = reduction_model.fit_transform(Z)

#trouver les meilleurs paramètres pour DBSCAN
def best_model_Dbscan(data, eps, min_samples):
    #Initialisation des paramètres (il doit avoir plus d'un cluster)
    i,j = 0,0
    while np.unique(DBSCAN(eps=eps[i], min_samples = min_samples[j]).fit_predict(data)).shape[0] < 2:
        i += 1
        j += 1
    best_model = DBSCAN(eps=eps[i], min_samples = min_samples[j])
    best_score = silhouette_score(data, best_model.fit_predict(data))
    #on teste toutes les combinaisons de paramètres
    for value in eps:
        for min_sample in min_samples:
            model = DBSCAN(eps=value, min_samples = min_sample)
            if np.unique(model.fit_predict(data)).shape[0] >= 2:
                score = silhouette_score(data, model.fit_predict(data))
                #on conserve celui qui a le meilleur score de Silhouette
                if score > best_score:
                   best_model = model
                   best_score = score
    return best_model
 
#va contenir les scores pour chaque valeur de K et pour chaque modèle : KMeans (col 0), GaussianMixture(col 1), 
#CAH (col 2) et DBSCAN (col 3)
K_silhouette_scores = np.zeros((19, 4))
#Modèle optimal pour DBSCAN    
eps = np.arange(0.1, 2, 0.1)
min_samples = range(3, 20)
Dbscan = best_model_Dbscan(Z, eps, min_samples)
print(Dbscan.get_params())
clusters = Dbscan.fit_predict(Z)
print("Nombre de clusters pour DBSCAN : ", clusters)
#Calcul du score de Silhouette pour DBSCAN
K_silhouette_scores[:, 3] = silhouette_score(Z, clusters) 
M = hierarchy.linkage(Z, method = 'average', metric= 'euclidean')
#Clustering pour différents nombres de classe
#Matrice pour contenir les inerties intra-classe pour chaque valeur de K et chaque modèle
inertia = []
for K in range(2,21):
    K_model_scores = []
    #Différents modèles
    models = {'KMeans': KMeans(n_clusters = K, random_state=0),'Gaussian': GaussianMixture(n_components = K, covariance_type = 'tied', random_state=0), 'Hierarchical' : M}
    index = 0
    for model_name, model in models.items():
        if(model_name == 'Hierarchical'):
            clusters = hierarchy.fcluster(model, t = K, criterion = 'maxclust')
        else:
            clusters = model.fit_predict(Z)
        #Calcul du score de silhouette obtenu pour ce modèle
        K_silhouette_scores[K-2, index] = silhouette_score(Z, clusters)
        #Pour KMeans, on calcule l'inertie pour chaque valeur de K => utile pour déterminer K (méthode du coude)
        if model_name == 'KMeans':
           inertia.append(model.inertia_)
        index += 1  


# #### Choix du nombre de clusters

# In[600]:


print("Score obtenu pour différentes valeurs de K pour chaque modèle \n", K_silhouette_scores)
print("\n Score pour chaque modèle moyenné sur les valeurs de K \n",K_silhouette_scores.mean(axis=0))
print("\n Score pour chaque valeur de K moyenné sur les différents modèles \n", K_silhouette_scores.mean(axis=1))
#On affiche l'évolution du score de Silhouette en fonction de K
plt.figure(figsize=(15,10))
plt.plot(range(2,21),  K_silhouette_scores.mean(axis=1))
plt.xlabel('K nombre de clusters')
plt.ylabel('Coefficient de Silhouette')
#Méthode du coude pour déterminer le nombre de clusters à retenir
plt.figure(figsize=(15,10))
plt.plot(range(2, 21), inertia)
plt.xlabel('K nombre de clusters')
plt.ylabel('Inertie inter-cluster')
plt.show()
#Dendrogramme
plt.figure(figsize=(15,10))
hierarchy.dendrogram(M)
plt.show()


# #### Analyse des clusters pour K = 17

# In[645]:


CAH_model = hierarchy.linkage(Z, method = 'average', metric= 'euclidean')
clusters = hierarchy.fcluster(M, t = 17, criterion = 'maxclust')
data['clusters_ACP'] = clusters
print(data.describe())
#statistiques par cluster
print(data.groupby('clusters_ACP').mean())


# #### Visualisation des clusters dans le plan principal

# In[649]:


plt.figure(figsize=(20,15))
plt.scatter(Z[:,0], Z[:,1], c=clusters)
plt.title('Visualisation des clisters dans le plan principal')
plt.xlabel('Composante principale 1')
plt.ylabel('Composante principale 2')


# #### Comparaison des partitions obtenues avec et sans ACP

# In[651]:


print(adjusted_rand_score(data['clusters'], data['clusters_ACP']))

