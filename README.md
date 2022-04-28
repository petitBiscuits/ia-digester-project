# Time Series Datasets
## Introduction
Ces trois datasets proviennent d'un projet effectué par l'équipe de RaD de la HE-Arc. Ils sont tous les trois des séries temporelles.

L'objectif principal avec ces données est de prédire une valeure dans le future. C'est une tâche de régression, plus d'information sur les metrics à utiliser ici: https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics

Ce document décrit ces trois datasets et explique le ou les objectifs pour chacuns.

Tous ces fichiers sont des .csv lisiblent facilement avec Pandas.
Tip: pandas.read_csv(filepath, index_col=0, parse_dates=True) pour les datasets avec une date comme index principal.

## Digester
Un digester est une usine qui produit du papier ou carton à partir du bois.

### Description des données
Nom du fichier: digester_data.csv

Il y a environ un mois de données dans ce fichier (5968 lignes). Les données sont enregistrées toutes les 10 minutes.

Il y a 37 colonnes dans ce dataset. Ce sont toutes des mesures provenant de capteurs dans l'usine.
La plus improtante est Blow Kappa qui mesure la qualité de la production.
L'objectif est de prédire ce Blow Kappa en utilisant les autres capteurs.