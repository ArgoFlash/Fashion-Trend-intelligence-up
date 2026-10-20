# Autoévaluation — Projet 2

État du 20/10/2026, d'après la fiche OpenClassrooms fournie.
Une case vide indique une action non terminée, et non une réussite supposée.

## Étape 1 — Environnement

- [x] Répertoire du projet créé.
- [x] Python 3.11 utilisé.
- [x] Environnement virtuel créé pour la validation.
- [x] Requests, Pillow, Matplotlib, Numpy et python-dotenv installés.
- [ ] Éditeur personnel configuré par l'utilisateur : non vérifié.

L'environnement de validation se trouve dans `work/venv` à côté du dossier
de livraison. Il n'est pas transportable et n'est pas inclus dans l'archive.
Le README permet de le recréer avec les versions figées.

## Étape 2 — Token Hugging Face

- [x] Token fourni par l'utilisateur et reconnu par `whoami-v2`.
- [x] Secret conservé dans `.env`, exclu de Git et de l'archive.
- [x] Validité testée par une requête réelle.
- [ ] Appel d'inférence autorisé avec crédits disponibles : HTTP 402 observé.

La portée détaillée du token n'a pas été auditée. Il doit disposer de
l'autorisation Inference Providers. Remplacer le token partagé dans le chat
après le projet.

## Étape 3 — Données

- [x] Archive officielle téléchargée.
- [x] 50 images et 50 masques organisés dans `asset/IMG` et `asset/Mask`.
- [x] Chargement, dimensions et identifiants de classe validés.
- [x] Appariement par identifiant et non par ordre de tri.

## Étape 4 — Script

- [x] Endpoint serverless actuel configuré pour **segformer_b3_clothes**.
- [x] Envoi PNG avec Requests, token externe et délais bornés.
- [x] Décodage et validation des masques testés hors réseau.
- [x] Format et taille contrôlés avant envoi.
- [x] Gestion des 429, erreurs serveur, timeout, refus d'accès et crédits.
- [x] Code commenté et tests de comportement.
- [ ] Réponse de segmentation réelle obtenue : bloquée par HTTP 402.

La fiche PDF mentionne encore `segformer_b2_clothes`. Le projet actualisé
et le notebook imposent `sayeed99/segformer_b3_clothes` : c'est ce modèle
qui a été retenu.

## Étape 5 — Analyse

- [x] Générateur de vues original / annotation / prédiction / superposition.
- [x] Palette cohérente entre classes et images.
- [x] Méthode de validation avec ensembles disjoints et métriques définies.
- [x] Rapport sur les limites et la montée en charge.
- [x] Estimation de coût sur 30 jours / 500 000 photos, avec hypothèses.
- [x] Présentation synthétique suivant le support fourni.
- [ ] 50 images segmentées via l'API.
- [ ] Visualisations des prédictions réelles produites.
- [ ] IoU / Dice réels analysés et présentation complétée avec les résultats.

**Le projet ne doit pas être présenté comme une évaluation terminée du modèle.**
À la demande de l'utilisateur, la livraison s'arrête aux fichiers fonctionnels
et tests hors API, sans lancer d'appels payants supplémentaires.
