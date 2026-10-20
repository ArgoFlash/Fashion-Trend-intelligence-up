# Bilan ModeTrends

Modèle : `sayeed99/segformer_b3_clothes`. Exécution via HF Inference.
Couverture : 0 succès sur 1 images demandées.
Objectif complet : 50 images. Une tentative refusée ne constitue pas une segmentation.

| Population | mIoU toutes classes | mIoU vêtements/accessoires | Dice macro |
|---|---:|---:|---:|
| Lot exploratoire | non mesuré | non mesuré | non mesuré |
| Train réservé | non mesuré | non mesuré | non mesuré |
| Validation | non mesuré | non mesuré | non mesuré |
| Test réservé | non mesuré | non mesuré | non mesuré |

## Interprétation et protocole

IoU = intersection / union. Dice = 2 × intersection / somme des surfaces.
Les matrices de confusion sont cumulées pixel par pixel avant la moyenne
des classes. Les classes absentes de la référence ET de la prédiction sont
exclues de la moyenne, et restent null dans le JSON.
Le fond peut gonfler l'accuracy : la mIoU vêtements est présentée séparément.
Les échecs API sont comptés dans la couverture, jamais transformés en prédictions.

Les 50 paires ont été réparties à graine fixe 42 : 30 train, 10 validation,
10 test. Aucun réentraînement n'est effectué. Les 30 images « train » ne
servent que de réserve pour une éventuelle adaptation future. Toute décision
de réglage doit être prise sur validation avant un test final.
Le bilan global est exploratoire. Le jeu fourni provient de human_parsing_dataset
(ATR), également cité comme source d'entraînement du modèle : le recouvrement
avec ses données d'entraînement n'est pas exclu. Ces scores ne prouvent donc
pas une généralisation sur de nouvelles publications Instagram. Il faut un
test indépendant, séparé par personne et par publication, avant production.
Les classes couvrent vêtements, accessoires et parties du corps. Il s'agit
de segmentation sémantique : deux personnes portant la même classe ne sont
pas distinguées comme instances.

## Coût pour 500 000 images / 30 jours

Volume moyen : 16 667 images/jour, soit 0,193 image/seconde.
HF Inference facture le temps de calcul multiplié par le tarif du matériel.
Le tarif unitaire de ce déploiement n'est pas publié dans les sources consultées.
La latence réseau mesurée ne permet pas de déduire la facture.

Scénarios de sensibilité à 0,50 USD/heure **supposé**, avec 5 % d'appels
supplémentaires facturés : 0,5 s/image → 36,46 USD ; 2 s → 145,83 USD ;
5 s → 364,58 USD. Ce sont des hypothèses, pas un devis Hugging Face.
Un endpoint dédié actif 720 heures au même tarif supposé coûterait 360 USD,
mais sa capacité et sa tarification doivent être vérifiées séparément.
Ces montants excluent taxes, stockage et crédits éventuels.

Pour une estimation fiable : relever le coût d'un lot suffisamment grand dans
la facturation HF, puis calculer 500 000 × coût du lot / nombre de succès.
Les erreurs et réessais du lot seront ainsi inclus. Ne pas multiplier encore
par les réessais si le coût du lot les contient déjà.

Sources consultées le 20/10/2026 :
- https://huggingface.co/docs/inference-providers/pricing
- https://huggingface.co/docs/inference-providers/tasks/image-segmentation
- https://huggingface.co/sayeed99/segformer_b3_clothes
