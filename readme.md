# Time Series Forecasting

## Objectif du projet
Ce projet vise à prévoir les ventes futures à partir d’un historique de données de vente journalières. L’objectif est de comprendre les tendances et la saisonnalité du comportement commercial afin d’aider à la planification logistique, les prévisions de stock et l’optimisation des ressources.

Le dataset utilisé est le fichier `train.csv`, qui contient des ventes journalieres pour plusieurs magasins et produits, avec les colonnes suivantes :

- `date`
- `store`
- `item`
- `sales`

## Données et contexte
Le projet exploite une série temporelle quotidienne sur plusieurs années. Les étapes principales de l’analyse sont :

1. Chargement du dataset et conversion de la colonne `date` en type datetime.
2. Agrégation des ventes par jour pour obtenir la série globale de ventes totales.
3. Analyse visuelle de la tendance et des variations saisonnières.
4. Modélisation avec Prophet pour prévoir les 90 prochains jours.
5. Evaluation des performances du modèle via MAE et MAPE.

## Stack technique
- Python 3.13
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Prophet

## Workflow du projet

### 1. Analyse exploratoire
Une première étape consiste à inspecter les données, vérifier leur structure et visualiser les ventes quotidiennes. La série agrégée par date permet d’observer la dynamique globale sur le temps.

### 2. Préparation pour la modélisation
Le modèle est construit sur un dataframe Prophet adapté :

- `ds` : date
- `y` : total des ventes par jour

La séparation des données est faite de la manière suivante :

- Entraînement : toutes les observations avant la dernière période de 90 jours
- Test : 90 derniers jours, utilisés pour évaluer la qualité de la prévision

### 3. Modélisation avec Prophet
Le modèle est configuré avec une saisonnalité annuelle activée et sans saisonnalité quotidienne.

```python
model = Prophet(yearly_seasonality=True, daily_seasonality=False)
model.fit(train)

future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)
```

### 4. Évaluation des performances
Le projet compare les prédictions du modèle aux vraies ventes sur la fenêtre de test à l’aide de :

- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)

## Résultats observés
Les analyses et les prévisions montrent que la série présente :

- une tendance générale positive,
- des variations saisonnières marquées,
- des pics sur certaines périodes de l’année,
- une bonne capacité du modèle à suivre les mouvements principaux de la demande.

Le notebook du projet affiche notamment :

- les ventes totales quotidiennes,
- la comparaison entre réalité et prédiction sur les 3 derniers mois,
- les composantes de tendance et de saisonnalité du modèle.

## Structure du dépôt
- `train.csv` : données historiques de ventes
- `Untitled.ipynb` : notebook d’exploration, modélisation et évaluation
- `main.py` : point d’entrée minimal du projet
- `pyproject.toml` : configuration Python et dépendances du projet

## Installation
```bash
pip install -e .
```

## Utilisation
Le projet est surtout exploité via le notebook Jupyter dans `Untitled.ipynb`. Il contient toutes les étapes de nettoyage, visualisation, modélisation et validation.

## Conclusion
Le projet illustre une démarche classique de forecasting de séries temporelles appliquée au retail, avec un modèle Prophet robuste pour capturer les tendances et les effets saisonniers sans nécessiter une préparation trop complexe des variables.

