# Détection de fraude bancaire

Un projet de machine learning supervisé qui apprend à repérer les transactions frauduleuses
à partir d'un historique de paiements. Le modèle analyse une transaction (montant, soldes avant
et après, type) et estime la probabilité qu'il s'agisse d'une fraude.

Projet Personnel.

## Les données

J'utilise PaySim, un jeu de données issu d'un simulateur de paiements mobiles : environ
6,3 millions de transactions, dont seulement 0,13 % de fraudes.

Le fichier fait près de 500 Mo, il n'est donc pas versionné ici. Pour le récupérer :
télécharger `paysim.csv` depuis [Hugging Face](https://huggingface.co/datasets/theman10/paysim)
et le placer dans `data/paysim_real.csv`.

Petite remarque : j'avais d'abord pris une version re-synthétisée de PaySim, mais l'exploration
a montré qu'elle n'avait aucun signal exploitable (les fraudes y étaient placées presque au hasard).
Je suis passé au vrai PaySim après avoir vérifié que les variables séparaient bien la fraude.

## La démarche

1. Exploration (EDA) pour comprendre les données et repérer les indices de fraude.
2. Préparation : nettoyage, puis création de variables utiles (notamment "le montant vide-t-il
   tout le compte", qui s'est révélée le signal le plus fort).
3. Séparation entraînement / test.
4. Entraînement d'une régression logistique.
5. Évaluation sur des données jamais vues.

Tout le détail, avec les explications, est dans le notebook `notebooks/fraude.ipynb`.

## Les résultats

Sur le jeu de test, le modèle attrape 2 453 fraudes sur 2 464 (recall de 99,6 %) avec seulement
22 fausses alertes (precision de 99,1 %). L'accuracy n'est pas une bonne mesure ici vu le fort
déséquilibre, je me suis donc appuyé sur la precision, le recall et la matrice de confusion.

Ces scores élevés tiennent beaucoup à la variable "vide tout le compte", presque parfaite dans
PaySim. Sur des fraudes réelles, plus subtiles, on n'atteindrait pas ce niveau.

## Lancer le projet

Installer les dépendances :

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Ouvrir le notebook :

```bash
jupyter notebook notebooks/fraude.ipynb
```

Lancer l'application de démo (on entre une transaction, le modèle répond) :

```bash
streamlit run app.py
```

## Structure

```
detection-fraude/
├── data/                     les données (à télécharger, voir plus haut)
├── notebooks/
│   └── fraude.ipynb          l'analyse complète, étape par étape
├── app.py                    l'application de démo (Streamlit)
├── modele_logistique.joblib  le modèle entraîné
├── scaler.joblib             la mise à l'échelle des variables
└── requirements.txt
```

## Outils

Python, pandas, scikit-learn, Streamlit, Jupyter.
