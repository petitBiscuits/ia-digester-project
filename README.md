# Rapport Digester - Izzo, Frossard

Projet final du cours d'IA basé sur le jeu de données "Digester". Le but du projet est d'entraîner plusieurs modèles tels qu’un "Neural Network", "Decision Tree" et "Support Vector" basé sur un problème de régression. Ils sont créés pour prédire la valeur "Blow Kappa" à l'aide des valeurs comprises dans le jeu de données.

## Exploration du jeu de données

L'utilisation de "Pandas" à permis de lire le fichier "csv" et afficher les données, dans le but de visualiser es caractéristiques du jeu de données. La fonction "read_cvs" lit le fichier et créer un "dataframe" contenant toutes les valeurs et avec la fonction "describe", cela affiche un tableau contenant des résultats statistiques sur les différentes statistiques, tels que le nombre d'entré, la moyenne et les quartiles. Cette analyse démontre que le jeu de données contient que des valeurs et une caractéristique qui est l'heure de la prise des mesures.

### Extraction des caractéristiques
Dans la continuité de l'exploration des données, il est intéressant d'afficher la corrélation entre les caractéristiques de notre jeu de données. Sur la "Heatmap", aucune caractéristique ne montre une forte corrélation avec la caractéristique "Blow Kappa" sauf lui-même. C'est pour cela qu’aucune caractéristique n’est retirée du jeu de données.

![](https://i.imgur.com/Yu8lTXk.png)


## Séparation des données

Pour les valeurs d'entraînements et de test. Une séparation de 80% du jeu de données est pour la partie d'entraînement et les 20% restant pour la partie de test. Cette séparation est effectuée dans les trois modèles.

## Baseline
Pour tous les modèles, un modèle dit "Baseline" a été réalisé. Il prédit tout le temps que la valeur rechercher du "Blow Kappa" est la moyenne des valeurs du "Blow Kappa" compris dans le jeu de donnée. Ce modèle permet de donner un modèle sur lequel se baser pour comparer la justesse des réels modèles créés.

## Neural Network
L'utilisation de "Keras" a permis de réaliser un réseau de neurones pour un problème de régression.

### Caractéristiques utilisées
Dans le cas du réseau de neurones, l'utilisation ou pas du "Blow Kappa" dans les caractéristiques donne des résultats proches. Mais le meilleur résultat du modèle est quand les valeurs du "BLow Kappa"sont comprises dans les caractéristiques.

### Window
"Keras" propose une fonction "timeseries_dataset_from_array" qui est utilisé pour des problèmes avec des valeurs dans le temps. Cela permet de créer des fenêtres contenant une suite d'entrées du jeu de données qui se suivent dans le temps. Sachant que notre jeu de données a sauvegardé des valeurs tout les 10 minutes, cela en fait six par heures. Donc voici les paramètres choisis dans la fonction : 
* sequence_length = 12 : contiens-les 120 dernières heures
* sampling_rate = 6 : créé une fenêtre toutes les prochaines heures

```python=
dataset_train = tf.keras.preprocessing.timeseries_dataset_from_array(
    X_train,
    y_train,
    sequence_length=12,
    sampling_rate=6,
    batch_size=256,
)

dataset_test = tf.keras.preprocessing.timeseries_dataset_from_array(
    X_test,
    y_test,
    sequence_length=sequence_length,
    sampling_rate=step,
    batch_size=batch_size,
)
```

### Model

Voici le modèle retenu pour le réseau de neurones. Pour commencer une couche "LSTM" d'une taille en sortie de 64, qui permet de se rappeler des valeurs précédentes. Ensuite, six couches de "Dense" viennent compléter le modèle. En sortie des couches c'est à chaque fois un multiple de 2, commencent par 64 pour la couche "LSTM" et se termine avec la dernière couche de "Dense" avec une seule valeur qui est la valeur à prédire "Blow Kappa".
D'autres modèles ont été tester tel que, une couche "LSTM" en sortie 64, une couche de "Dense" en sortie 32 et une dernière chouche de "Dense" en sortie 1. Cependant, les résultats n'étaient pas concluants alors l'ajout de couche "Dense" s'est fait naturellement ce qui a affiné le modèle.

```
Model: "model_28"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_29 (InputLayer)       [(None, 12, 37)]          0         
                                                                 
 lstm_28 (LSTM)              (None, 64)                26112     
                                                                 
 dense_151 (Dense)           (None, 32)                2080      
                                                                 
 dense_152 (Dense)           (None, 16)                528       
                                                                 
 dense_153 (Dense)           (None, 8)                 136       
                                                                 
 dense_154 (Dense)           (None, 4)                 36        
                                                                 
 dense_155 (Dense)           (None, 2)                 10        
                                                                 
 dense_156 (Dense)           (None, 1)                 3         
                                                                 
=================================================================
Total params: 28,905
Trainable params: 28,905
Non-trainable params: 0
_________________________________________________________________
```

### Résultat
Après plusieurs entraînements, le réseau de neurones donne de bons résultats avec une "mae" de 0.8 qui comparé à la "baseline" est de 1.13.

## Decision Tree Regression
Avec "sklearn" l'utilisation de "DescisionTreeRegressor" permet la création d'un arbre de décision pour un problème de régression.

>Réalisé par Izzo Valentino

### Caractéristiques utilisées
Dans le cas de l'arbre de décision, l'utilisation ou pas du "Blow Kappa" dans les caractéristiques donne des résultats éloignés. Mais le meilleur résultat du modèle est quand dans les valeurs du "BLow Kappa" sont comprises dans les caractéristiques.

### GridSearch
GridSearchCV permet d'estimer les meilleurs paramètres qu'on lui donne. Voici la liste des paramètres testés :
* max_depth[3, 5, 10] : le nombre maximal de couches de l'arbre
* splitter[best, random] : la stratégie utilisée pour choisir le fractionnement à chaque nœud.
* min_samples_split[2, 5, 10] : Le nombre minimum d'échantillons requis pour diviser un noeud interne.
* min_samples_leaf[1, 2, 3, 4, 5] : Le nombre minimum d'échantillons requis pour être à un noeud feuille.
* max_features[auto, sqrt, log2, None] : Le nombre de caractéristiques à prendre en compte lors de la recherche du meilleur split.
* random_state[0, 1, 2, 3, 4, 5] : Contrôle l'aléatoire de l'estimateur.
* min_impurity_decrease[0.0, 0.1, 0.2, 0.3] : Un nœud sera divisé si cette division induit une diminution de l'impureté supérieure ou égale à cette valeur.

Cette recherche a duré 86 minutes.

### Hyperparametre
Voici les paramètres choisis en fonction du GridSearch :
* max_depth : 5
* splitter : random
* min_samples_split : 5
* min_samples_leaf : 1
* max_features : sqrt
* random_state : 0
* min_impurity_decrease : 0.0
* criterion: absolute_error

### Résultat
Après l'affichage de quelques valeurs de prédiction, les nombres sont proches. Cependant, la valeur de la "Mean absolute error" est moins bonne que la valeur de la "baseline". Avec un "mae" de 1.50 et un "mae" de 1.13 pour la "baseline" calculer avec la moyenne. Avant l'utilisation de GridSearch pour trouver les meilleurs paramètres, les valeurs "mae" étaient proches du résultat actuel. Aucune configuration essayée n’arrive à être meilleure que la moyenne.

À la fin l'arbre de décision est affiché avec GraphViz pour un rendu visuel de l'arbre.

## Support Vector Regression

Support vector machines (SVM) sont utiles pour l'apprentissage supervisé. Ils sont utilisés pour la classification et dans notre cas la régression.

>Réalisé par Frossard Loïc
### GridSearch

La variation des hyper paramètres suivants a été testée:


- kernel['rbf'] = Spécifie le type de kernel utiliser dans l'algorithme. Choisis Rbf, car dans les tests les autres donnaient des valeurs beaucoup trop grandes.
- C[1, 10, 100] = Un paramètre de régularisation
- epsilon[0.01, 0.05, 0.1, 0.5, 1] = La valeur de pénalité associée a la fonction de loss.
- gamma[0.01, 0.1, 1] = Kernel coef


### Hyperparametre choisit

Les meilleurs paramètres dont le GridSearch a trouvé sont les valeurs suivantes : 

- kernel ['rbf']
- gamma [0.01]
- epsilon [0.01]
- C[100]

Les valeurs choisies sont toutes sur les extrêmes opposés. Donc ce qu'il faudrait faire s'est rajouté des valeurs aux tables par exemple le C [1,10,100, 500, 1000]. Mais par manque de temps cela n'a pas été fait, car le train des données prend beaucoup de temps une moyenne de 20 min environ.

### Result

Lors des tests du GridSearch la prédiction du SVR de bonnes valeurs plutôt différentes. Mais après un certain temps, la fonction de prédiction retourne sur toutes les prédictions la même valeur.

![](https://i.imgur.com/AJia6fL.png)


L'hypothèse que Python enregistre l’entraînement a été pensée. Donc la suppression des variables créées dans le notebook a été réalisée, mais cela n'a rien changé. Aucune idée de comment cela ce fait

avec le problème précédent le MAE du SVM est légèrement au-dessus du MAE réalisée avec la moyenne.

Donc la Comparaison des deux MAE.

```
SVR : 1.2873748408615433
Base Line :  1.1396043875364883
```
