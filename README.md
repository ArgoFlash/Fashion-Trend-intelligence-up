# ModeTrends — Requêtez des services IA

Projet OpenClassrooms : segmentation vestimentaire avec
`sayeed99/segformer_b3_clothes`, via **HF Inference serverless**.

## État de la livraison

Les 50 images et 50 annotations officielles sont incluses. Le code, le notebook,
les tests, le calcul de coût et le support de présentation sont fournis.
Le token a été accepté lors du test réel, mais l'inférence a renvoyé **HTTP 402**
(crédits insuffisants). À votre demande, les appels payants ont été arrêtés.
**Aucune prédiction ni performance du modèle n'est présentée comme mesurée.**
Les tests logiciels emploient des données synthétiques explicitement séparées.

## Installation

Version de référence : Python 3.11. Un environnement virtuel neuf est recommandé.

```powershell
cd chemin\vers\modetrends
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
Copy-Item .env.example .env
```

Ne recopiez pas `.env.example` si votre `.env` est déjà renseigné.
Ouvrez `.env` dans votre éditeur et placez le token après `HF_TOKEN=`.
N'écrivez pas la clé dans une cellule du notebook ou dans une commande.
Le fichier `.env`, les caches et l'environnement virtuel sont exclus de Git.
L'archive livrée ne contient pas le secret.

Le token doit autoriser les appels Inference Providers. Le compte doit disposer
des crédits nécessaires. Le script ne crée ni abonnement ni infrastructure.
Il n'effectue aucun appel tant que vous ne lancez pas `run` ou la cellule
d'inférence du notebook avec `EXECUTER_API = True`.

## Commandes

```powershell
# Vérification hors réseau des 50 paires et création des ensembles
.\.venv\Scripts\python.exe modetrends.py audit

# Tests logiciels hors réseau
.\.venv\Scripts\python.exe -m pytest -q

# Après approvisionnement du compte : un premier essai réel
.\.venv\Scripts\python.exe modetrends.py run --limit 1

# Puis les 50 images (les succès déjà enregistrés sont réutilisés)
.\.venv\Scripts\python.exe modetrends.py run --limit 50

# Recalcul hors réseau depuis les résultats existants
.\.venv\Scripts\python.exe modetrends.py report

# Scénarios de coût, sans requête réseau
.\.venv\Scripts\python.exe modetrends.py cost
```

Sous macOS/Linux, utilisez `.venv/bin/python` à la place.
Le notebook `huggingface_api_cloth_seg.ipynb` s'ouvre dans VS Code avec le
kernel Python de `.venv`. Exécutez les cellules dans l'ordre.
Il reprend les six étapes pédagogiques du notebook source.

## Fichiers

- `modetrends.py` : préparation, API Requests, masques, évaluation et rapport.
- `huggingface_api_cloth_seg.ipynb` : parcours commenté et résultats enregistrés.
- `asset/IMG` et `asset/Mask` : jeu officiel, associé par identifiant numérique.
- `tests/` : métriques sur cas calculables, masques, retries, erreurs et données.
- `results/run.json` : journal des essais, sans identité ni token.
- `results/splits.json` : répartition reproductible 30 / 10 / 10.
- `results/metrics.json` : métriques par classe et par ensemble ; null si non mesurées.
- `results/rapport.md` : analyse, limites et chiffrage approximatif.
- `results/index.html` : galerie, consultable en ouvrant le fichier dans un navigateur.
- `results/raw`, `results/masks`, `results/figures` : remplis après une inférence réussie.
- `AUTOEVALUATION.md` : suivi des attendus.

## Choix techniques et limites

L'adresse du notebook d'origine a été remplacée par le routeur actuel
`https://router.huggingface.co/hf-inference/models/sayeed99/segformer_b3_clothes`.
Les images sont converties en RGB PNG et limitées à 1 024 pixels sur le grand
côté. Les masques sont ramenés à la taille originale au plus proche voisin.
La comparaison utilise les pixels de la vérité terrain, jamais une image colorisée.

Les requêtes possèdent des délais de connexion/lecture, trois essais au maximum
pour 429 et certaines erreurs serveur, et une pause entre les images. Les refus
d'accès ou de crédit arrêtent le lot. Un relancement reprend les succès en cache,
identifiés par le contenu envoyé et l'endpoint. Le modèle distant n'est pas
figé à une révision : conserver les réponses brutes permet d'auditer ce test.

Le découpage est fixé avant l'analyse. Aucun entraînement n'est demandé pour
ce modèle préentraîné. Le lot de 50 images est une exploration ; le test réservé
doit rester indépendant des choix effectués sur la validation. Les données
proviennent d'ATR, utilisé aussi pour entraîner le modèle : un jeu indépendant
est nécessaire avant une conclusion de généralisation.

Les 18 classes incluent le corps et le fond. La mIoU vêtements/accessoires
sépare l'objectif métier de l'accuracy dominée par le fond. Les métriques
ignorent la classe 255 et les classes absentes à la fois en référence et prédiction.

## Coût

Le calcul demandé est inclus avec **hypothèses explicites**, pas un tarif inventé.
HF Inference facture le calcul utilisé. La durée HTTP inclut le réseau et les
réessais et ne doit pas être assimilée au calcul facturé.
Pour obtenir un budget exploitable, relever le coût réel d'un lot dans la
facturation, puis extrapoler à 500 000 images.

## Sources

- Énoncé : https://openclassrooms.com/fr/users/9431275/paths/2464/projects/3158/4337-exercice---requetez-des-services-ia
- Modèle : https://huggingface.co/sayeed99/segformer_b3_clothes
- API : https://huggingface.co/docs/inference-providers/tasks/image-segmentation
- Tarification : https://huggingface.co/docs/inference-providers/pricing

Les ressources pédagogiques et données restent attribuées à leurs auteurs.
Ne publiez pas automatiquement les images du jeu sur un dépôt public.
