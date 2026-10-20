"""Fashion Trend Intelligence : expérimentation reproductible de segmentation.

Ce module complète le notebook OpenClassrooms : secret externe, API actuelle,
reprise des calculs, validation des masques, métriques et rapport consultable.
Les annotations ne sont jamais utilisées pour construire les prédictions.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import io
import json
import os
from pathlib import Path
import random
import time

# Garder les caches dans le projet évite de modifier le profil Windows.
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mpl-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from PIL import Image
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
MODEL = "sayeed99/segformer_b3_clothes"
API_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL}"
LABELS = ["Background", "Hat", "Hair", "Sunglasses", "Upper-clothes",
          "Skirt", "Pants", "Dress", "Belt", "Left-shoe", "Right-shoe",
          "Face", "Left-leg", "Right-leg", "Left-arm", "Right-arm", "Bag", "Scarf"]
CLASS_MAPPING = {name: i for i, name in enumerate(LABELS)}
CLOTHES = [1, 3, 4, 5, 6, 7, 8, 9, 10, 16, 17]
COLORS = np.array([
    [30, 34, 42], [255, 196, 0], [151, 93, 43], [199, 0, 255],
    [238, 65, 75], [0, 190, 190], [66, 190, 74], [50, 105, 240],
    [150, 55, 140], [255, 139, 0], [185, 100, 0], [236, 188, 150],
    [110, 155, 160], [90, 120, 145], [235, 145, 170], [195, 110, 140],
    [40, 180, 250], [255, 80, 180]], dtype=np.uint8)


def write_json(path, data):
    """Écriture atomique : une interruption ne laisse pas un cache à moitié écrit."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def dataset_pairs(asset=ROOT / "asset"):
    """Associer par identifiant, jamais par position dans deux listes triées."""
    images = sorted((Path(asset) / "IMG").glob("image_*.png"),
                    key=lambda p: int(p.stem.split("_")[-1]))
    if not images:
        raise ValueError("Aucune image dans asset/IMG.")
    pairs = []
    for image in images:
        mask = Path(asset) / "Mask" / image.name.replace("image_", "mask_", 1)
        if not mask.exists():
            raise ValueError(f"Annotation absente : {mask.name}")
        with Image.open(image) as im, Image.open(mask) as gt:
            array = np.asarray(gt)
            if gt.size != im.size or array.ndim != 2:
                raise ValueError(f"Dimensions ou format invalides : {mask.name}")
            if not np.isin(array, list(range(18)) + [255]).all():
                raise ValueError(f"Classes inconnues : {mask.name}")
        pairs.append((image, mask))
    return pairs


def make_splits(pairs):
    """Découpage fixé AVANT les résultats, destiné à un futur ajustement.

    Le modèle est déjà entraîné : aucun entraînement n'est effectué ici.
    Le bilan des 50 images est exploratoire, le test réservé est distingué.
    Les identités des personnes ne sont pas fournies : absence de fuite entre
    personnes non garantie, ni absence de recouvrement avec l'entraînement ATR.
    """
    ids = [p.stem for p, _ in pairs]
    random.Random(42).shuffle(ids)
    a, b = int(len(ids) * .6), int(len(ids) * .8)
    return {"train": ids[:a], "validation": ids[a:b], "test": ids[b:]}


class APIError(RuntimeError):
    pass


class HFClient:
    def __init__(self, token, session=None, sleep=time.sleep, max_attempts=3):
        if not token:
            raise APIError("HF_TOKEN absent. Renseigner .env, puis relancer.")
        self.session = session or requests.Session()
        self.headers = {"Authorization": f"Bearer {token}"}
        self.sleep = sleep
        self.max_attempts = max_attempts

    def authenticate(self):
        response = self.session.get("https://huggingface.co/api/whoami-v2",
                                    headers=self.headers, timeout=(10, 30))
        if response.status_code != 200:
            raise APIError(f"Authentification refusée (HTTP {response.status_code}).")
        # Ne pas enregistrer l'identité du compte, le token ou les en-têtes.
        return True

    def segment(self, payload):
        """Réessais bornés pour 429/5xx et problèmes réseau.

        Un timeout peut survenir après traitement serveur : un réessai peut
        donc coûter un appel supplémentaire. Le nombre d'essais est enregistré.
        """
        start = time.perf_counter()
        for attempt in range(1, self.max_attempts + 1):
            delay = min(2 ** attempt, 30)
            try:
                response = self.session.post(
                    API_URL, headers={**self.headers, "Content-Type": "image/png"},
                    data=payload, timeout=(10, 120), allow_redirects=False)
            except (requests.Timeout, requests.ConnectionError):
                if attempt == self.max_attempts:
                    raise APIError("Délai dépassé ou connexion indisponible.") from None
            else:
                if response.status_code == 200:
                    try:
                        result = response.json()
                    except ValueError:
                        raise APIError("La réponse API n'est pas du JSON.") from None
                    if not isinstance(result, list) or not result:
                        raise APIError("L'API n'a renvoyé aucun segment valide.")
                    return result, time.perf_counter() - start, attempt
                if response.status_code not in (429, 500, 502, 503, 504):
                    explanations = {
                        401: "Token invalide", 402: "Crédits d'inférence insuffisants",
                        403: "Permission Inference Providers manquante",
                        404: "Modèle ou endpoint indisponible",
                        410: "Endpoint retiré"}
                    raise APIError(f"HTTP {response.status_code} : "
                                   f"{explanations.get(response.status_code, 'appel refusé')}.")
                if attempt == self.max_attempts:
                    raise APIError(f"HTTP {response.status_code} après {attempt} essais.")
                try:
                    delay = min(max(float(response.headers.get("Retry-After", delay)), 0), 60)
                except ValueError:
                    pass
            self.sleep(delay)
        raise APIError("Aucune réponse exploitable.")


def prepare_image(path, max_side=1024):
    """PNG RGB sans métadonnées, ratio conservé, aucune déformation.

    Les images du jeu sont déjà orientées comme leurs masques. On ne modifie
    donc pas l'orientation EXIF indépendamment des annotations.
    """
    if max_side < 1:
        raise ValueError("max_side doit être positif.")
    with Image.open(path) as source:
        image = source.convert("RGB")
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue(), image.size


def create_masks(results, width, height):
    """Décoder les masques binaires ; le fond ne doit pas écraser les vêtements.

    Pour un éventuel chevauchement de classes, le score du segment départage
    les pixels. Ce score n'est pas une probabilité indépendante par pixel.
    """
    combined = np.zeros((height, width), dtype=np.uint8)
    confidence = np.full((height, width), -np.inf)
    for result in results:
        label = result.get("label")
        if label not in CLASS_MAPPING:
            raise APIError(f"Classe API inconnue : {label!r}")
        try:
            decoded = base64.b64decode(result["mask"], validate=True)
            with Image.open(io.BytesIO(decoded)) as image:
                binary = np.asarray(image.convert("L").resize(
                    (width, height), Image.Resampling.NEAREST)) > 0
            score = float(result.get("score") if result.get("score") is not None else 1)
            if not np.isfinite(score):
                raise ValueError
        except (ValueError, KeyError, OSError, TypeError):
            raise APIError("Masque ou score API invalide.") from None
        class_id = CLASS_MAPPING[label]
        if class_id == 0:
            continue
        selected = binary & (score > confidence)
        combined[selected] = class_id
        confidence[selected] = score
    return combined


def confusion_matrix(truth, prediction):
    if truth.shape != prediction.shape:
        raise ValueError("Les masques doivent avoir les mêmes dimensions.")
    valid = truth != 255  # Pixels explicitement ignorés par l'annotation.
    if not np.isin(truth[valid], range(18)).all() or not np.isin(prediction, range(18)).all():
        raise ValueError("Identifiant de classe hors de [0, 17].")
    indices = 18 * truth[valid].astype(np.int64) + prediction[valid].astype(np.int64)
    return np.bincount(indices, minlength=324).reshape(18, 18)


def metrics(matrix):
    tp = np.diag(matrix).astype(float)
    actual, predicted = matrix.sum(1), matrix.sum(0)
    union = actual + predicted - tp
    iou = np.divide(tp, union, out=np.full(18, np.nan), where=union > 0)
    dice = np.divide(2 * tp, actual + predicted,
                     out=np.full(18, np.nan), where=(actual + predicted) > 0)
    # Une classe absente à la fois en référence et en prédiction est exclue.
    mean = lambda x: float(np.nanmean(x)) if np.isfinite(x).any() else None
    return {
        "miou": mean(iou), "miou_without_background": mean(iou[1:]),
        "miou_clothes": mean(iou[CLOTHES]), "macro_dice": mean(dice),
        "pixel_accuracy": float(tp.sum() / matrix.sum()) if matrix.sum() else None,
        "classes": [{"id": i, "label": label,
                     "iou": float(iou[i]) if np.isfinite(iou[i]) else None,
                     "dice": float(dice[i]) if np.isfinite(dice[i]) else None,
                     "reference_pixels": int(actual[i]),
                     "predicted_pixels": int(predicted[i])}
                    for i, label in enumerate(LABELS)]}


def visualize(image_path, truth, prediction, destination):
    with Image.open(image_path) as im:
        image = np.asarray(im.convert("RGB"))
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    safe_truth = np.where(truth == 255, 0, truth)
    overlay = image.copy()
    foreground = prediction > 0
    overlay[foreground] = (.55 * image[foreground] + .45 * COLORS[prediction[foreground]]).astype(np.uint8)
    for ax, picture, title in zip(axes, [image, COLORS[safe_truth], COLORS[prediction], overlay],
                                 ["Original", "Annotation", "Prédiction HF", "Superposition"]):
        ax.imshow(picture)
        ax.set_title(title)
        ax.axis("off")
    present = sorted(set(np.unique(safe_truth)) | set(np.unique(prediction)))
    fig.legend(handles=[Patch(color=COLORS[i] / 255, label=LABELS[i]) for i in present],
               loc="lower center", ncol=6, fontsize=8)
    fig.suptitle(image_path.stem)
    fig.tight_layout(rect=(0, .13, 1, .95))
    fig.savefig(destination, dpi=120)
    plt.close(fig)


def cost_scenarios(n=500_000, days=30, seconds=2.0, hourly_rate=.5, retry_rate=.05):
    """Hypothèses explicites, pas un tarif SegFormer observé."""
    if n <= 0 or days <= 0 or seconds <= 0 or hourly_rate < 0 or retry_rate < 0:
        raise ValueError("Hypothèses de coût invalides.")
    return {"images": n, "days": days, "images_per_day": n / days,
            "requests_per_second": n / (days * 86400),
            "assumed_billable_seconds_per_image": seconds,
            "assumed_usd_per_compute_hour": hourly_rate,
            "extra_billed_attempt_fraction": retry_rate,
            "estimated_usd": n * seconds / 3600 * hourly_rate * (1 + retry_rate),
            "dedicated_always_on_usd": days * 24 * hourly_rate,
            "warning": "Scénario hypothétique, hors taxes, stockage et crédits. "
                       "La latence HTTP n'est pas le temps de calcul facturé."}


def report(output, records, splits):
    """Recalculer les métriques depuis les masques conservés, sans nouvel appel."""
    matrices = {name: np.zeros((18, 18), dtype=np.int64)
                for name in ["all", "train", "validation", "test"]}
    rows, panels = [], []
    for record in records:
        if record["status"] != "ok":
            continue
        image_id = record["image"]
        truth = np.asarray(Image.open(ROOT / "asset/Mask" / image_id.replace("image_", "mask_")).convert("L"))
        pred = np.asarray(Image.open(output / "masks" / image_id))
        matrix = confusion_matrix(truth, pred)
        matrices["all"] += matrix
        split = next(key for key, values in splits.items() if Path(image_id).stem in values)
        matrices[split] += matrix
        scores = metrics(matrix)
        rows.append({"image": image_id, "split": split, "miou": scores["miou"],
                     "miou_clothes": scores["miou_clothes"],
                     "seconds": record["seconds"], "attempts": record["attempts"]})
        panels.append(f'<article><h2>{html.escape(image_id)}</h2>'
                      f'<img loading="lazy" src="figures/{Path(image_id).stem}.png"></article>')
    summary = {key: metrics(value) for key, value in matrices.items()}
    summary["coverage"] = {"requested": len(records), "successful": len(rows),
                           "failed": len(records) - len(rows)}
    summary["model"] = MODEL
    summary["endpoint"] = API_URL
    summary["latency_note"] = "Durée HTTP avec attentes et réessais, pas durée facturée."
    write_json(output / "metrics.json", summary)
    with (output / "per_image.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "split", "miou",
                                              "miou_clothes", "seconds", "attempts"])
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / "costs.json", [cost_scenarios(seconds=s) for s in [.5, 2, 5]])
    def percent(value):
        return "non mesuré" if value is None else f"{100 * value:.1f} %"
    text = f"""# Bilan ModeTrends

Modèle : `{MODEL}`. Exécution via HF Inference.
Couverture : {len(rows)} succès sur {len(records)} images demandées.
Objectif complet : 50 images. Une tentative refusée ne constitue pas une segmentation.

| Population | mIoU toutes classes | mIoU vêtements/accessoires | Dice macro |
|---|---:|---:|---:|
"""
    for name, label in [("all", "Lot exploratoire"), ("train", "Train réservé"),
                        ("validation", "Validation"), ("test", "Test réservé")]:
        m = summary[name]
        text += f"| {label} | {percent(m['miou'])} | {percent(m['miou_clothes'])} | {percent(m['macro_dice'])} |\n"
    text += """
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
"""
    (output / "rapport.md").write_text(text, encoding="utf-8")
    page = ("<!doctype html><html lang='fr'><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width, initial-scale=1'>"
            "<title>ModeTrends — résultats de segmentation</title>"
            "<style>body{font:17px system-ui;background:#f3f5f7;color:#172530;"
            "max-width:1400px;margin:40px auto;padding:20px}article{background:white;"
            "padding:20px;margin:20px 0;border-radius:12px}img{width:100%;height:auto}"
            "a{color:#065d83}</style><h1>ModeTrends · Segmentation vestimentaire</h1>"
            f"<p>{len(rows)} / {len(records)} images traitées. "
            f"mIoU vêtements : {percent(summary['all']['miou_clothes'])}.</p>"
            "<p>Résultats exploratoires. Recouvrement avec l’entraînement possible.</p>"
            "<p><a href='rapport.md'>Rapport</a> · <a href='metrics.json'>Métriques</a>"
            " · <a href='per_image.csv'>Détail par image</a></p>" + "".join(panels) + "</html>")
    (output / "index.html").write_text(page, encoding="utf-8")
    return summary


def run(limit=50, max_side=1024, pause=1.0):
    pairs = dataset_pairs()
    splits = make_splits(pairs)
    output = ROOT / "results"
    for folder in ["raw", "masks", "figures"]:
        (output / folder).mkdir(parents=True, exist_ok=True)
    write_json(output / "splits.json", splits)
    load_dotenv(ROOT / ".env", encoding="utf-8-sig")
    client = HFClient(os.getenv("HF_TOKEN"))
    client.authenticate()
    records = []
    for image, gt in pairs[:limit]:
        payload, sent_size = prepare_image(image, max_side)
        fingerprint = hashlib.sha256(payload + API_URL.encode()).hexdigest()
        cache = output / "raw" / f"{image.stem}.json"
        record = {"image": image.name, "status": "error", "fingerprint": fingerprint}
        try:
            previous = json.loads(cache.read_text("utf-8")) if cache.exists() else None
            if previous and previous.get("fingerprint") == fingerprint:
                response, seconds, attempts = previous["response"], previous["seconds"], previous["attempts"]
                cached = True
            else:
                response, seconds, attempts = client.segment(payload)
                cached = False
            with Image.open(image) as original:
                width, height = original.size
            prediction = create_masks(response, width, height)
            write_json(cache, {"fingerprint": fingerprint, "response": response,
                               "seconds": seconds, "attempts": attempts})
            Image.fromarray(prediction).save(output / "masks" / image.name)
            truth = np.asarray(Image.open(gt))
            visualize(image, truth, prediction, output / "figures" / f"{image.stem}.png")
            record.update(status="ok", seconds=seconds, attempts=attempts,
                          cached=cached, sent_size=sent_size)
            print(f"{image.name}: OK ({seconds:.2f}s, cache={cached})", flush=True)
            if not cached:
                time.sleep(pause)
        except (APIError, requests.RequestException, ValueError, OSError) as exc:
            # Les exceptions requests peuvent contenir une URL, jamais les headers.
            message = str(exc) if isinstance(exc, APIError) else type(exc).__name__
            record["error"] = message
            print(f"{image.name}: ÉCHEC — {message}", flush=True)
        records.append(record)
        write_json(output / "run.json", records)
        # Ne pas répéter 50 fois un refus d'accès ou un manque de crédits.
        if record["status"] == "error" and any(x in record.get("error", "") for x in ["HTTP 401", "HTTP 402", "HTTP 403", "HTTP 404", "HTTP 410"]):
            break
    summary = report(output, records, splits)
    if summary["coverage"]["failed"]:
        raise APIError("Exécution incomplète. Voir results/run.json, puis relancer.")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["audit", "run", "report", "cost"])
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--max-side", type=int, default=1024)
    args = parser.parse_args()
    if args.command == "audit":
        pairs = dataset_pairs()
        write_json(ROOT / "results/splits.json", make_splits(pairs))
        print(f"{len(pairs)} paires image/masque valides.")
    elif args.command == "run":
        if not 1 <= args.limit <= 50:
            parser.error("--limit doit être entre 1 et 50.")
        try:
            run(args.limit, args.max_side)
        except (APIError, requests.RequestException) as exc:
            parser.exit(1, f"{exc}\n")
    elif args.command == "report":
        output = ROOT / "results"
        report(output, json.loads((output / "run.json").read_text("utf-8")),
               json.loads((output / "splits.json").read_text("utf-8")))
    else:
        print(json.dumps([cost_scenarios(seconds=s) for s in [.5, 2, 5]], indent=2))


if __name__ == "__main__":
    main()
