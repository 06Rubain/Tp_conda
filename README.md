# Tp_conda
just un bon depart avec conda py
# Chapitre 2 : Chargement et Analyse des Données
# Chargement des Librairies
import numpy as np  
import pandas as pd  
# Chargement des Données
data = pd.read_csv("Data/studentscores.csv")  
data.head(4)  

# Analyse des Données
# Vérification des valeurs manquantes  
data.isnull().sum()  

# Informations sur les données  
data.info()  

# Valeurs maximales  
data['Hours'].max()  
data['Scores'].max()  

print('Max score:', data['Scores'].max())  
print('Min score:', data['Scores'].min())  


## Division des Données
# Variables Indépendantes et Dépendantes
x = data.iloc[:, :-1].values  
y = data.iloc[:, -1]  

# Séparation des Données en Train et Test
from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=40)  

# Division des Données
# Variables Indépendantes et Dépendantes
x = data.iloc[:,:-1].values
y = data.iloc[:,-1]

# Séparation des Données en Train et Test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=40)

# Construction du Modèle
# Entraînement du Modèle
from sklearn.linear_model import LinearRegression 
modele_lineaire = LinearRegression()
modele_lineaire.fit(x_train, y_train)

# Prédictions
y_scores_predict = modele_lineaire.predict(x_test)

# Évaluation du Modèle
from sklearn.metrics import mean_squared_error, r2_score 
print("MSE:", mean_squared_error(y_test, y_scores_predict))
print("R2:", r2_score(y_test, y_scores_predict))

# Visualisation des Résultats
import matplotlib.pyplot as plt 
plt.scatter(x_train, y_train, color="blue", label='train data')
plt.plot(x_test, y_scores_predict, color='red', label='ligne de la regression')
plt.legend()
plt.show()

# Prédiction d'une Nouvelle Valeur
modele_lineaire.predict([[2.5]])

# Comparaison des Résultats Prédits et Réels
resultats = pd.DataFrame({'y_score_real': y_test, 'y_scores_predict': y_scores_predict})
resultats



# Auteurs
Projet réalisé par [Rubain Ntita].

# Licence
Ce projet est sous licence MIT.
