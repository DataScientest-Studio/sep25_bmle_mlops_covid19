"""
Schémas de présentation en PNG (matplotlib), version lisible pour jury.
Design volontairement simple : gros blocs, peu de texte par bloc, fort contraste.
"""
import io

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

FACE = "#ffffff"
TEXT = "#0f172a"
SUBTEXT = "#475569"
BOX = "#eef2ff"
EDGE = "#4f46e5"
ARROW = "#4f46e5"
DPI = 120


def _make_canvas(width=16, height=8):
    fig, ax = plt.subplots(figsize=(width, height), facecolor=FACE, dpi=DPI)
    ax.set_facecolor(FACE)
    ax.axis("off")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    return fig, ax


def _box(ax, x, y, w, h, text, fs=12):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.04",
        facecolor=BOX,
        edgecolor=EDGE,
        linewidth=2.2,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=TEXT)


def _soft_panel(ax, x, y, w, h):
    panel = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.03",
        facecolor="#f8faff",
        edgecolor="#dbeafe",
        linewidth=1.0,
    )
    ax.add_patch(panel)


def _arrow(ax, x1, y1, x2, y2, dashed=False):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", lw=2.4, color=ARROW, linestyle="--" if dashed else "-"),
    )


def _to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=FACE, dpi=DPI)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def diagram_architecture() -> bytes:
    """Architecture globale (version epuree, lisible en soutenance)."""
    fig, ax = _make_canvas(16, 8)

    # Titre + sous-titre
    ax.text(0.8, 9.35, "Architecture globale du systeme", fontsize=20, weight="bold", color=TEXT)
    ax.text(0.8, 8.85, "Flux principal (haut) + boucle de feedback clinique (bas)", fontsize=12, color=SUBTEXT)

    # Bandeau principal (plus lisible visuellement)
    _soft_panel(ax, 0.6, 5.15, 14.8, 2.25)

    # Chaine principale (5 blocs, bien espaces)
    y = 5.95
    w = 2.35
    h = 1.25
    xs = [0.9, 3.9, 6.9, 9.9, 12.9]
    labels = [
        "Kaggle\n(Dataset)",
        "Pipeline\nDonnees",
        "Entrainement\nEfficientNetV2",
        "Modele\nversionne",
        "API\nFastAPI",
    ]
    for x, lb in zip(xs, labels):
        _box(ax, x, y, w, h, lb, 12)
    for i in range(len(xs) - 1):
        _arrow(ax, xs[i] + w, y + h / 2, xs[i + 1], y + h / 2)

    # Boucle clinique en bas (schema simple, sans chevauchement)
    _box(ax, 3.0, 2.25, 3.0, 1.15, "Feedback\nmedecin", 11)
    _box(ax, 7.0, 2.25, 3.6, 1.15, "Supabase\n(images_dataset)", 11)
    _box(ax, 11.4, 2.25, 2.8, 1.15, "App clinique", 12)

    # API -> App (connexion verticale courte, ne traverse pas les cadres)
    _arrow(ax, 14.1, 5.95, 12.8, 3.4)
    # App -> Supabase -> Feedback (gauche)
    _arrow(ax, 11.4, 2.82, 10.6, 2.82)
    _arrow(ax, 7.0, 2.82, 6.0, 2.82)
    # Feedback -> retrain (retour vers entrainement)
    _arrow(ax, 6.0, 3.35, 8.2, 5.95, dashed=True)
    ax.text(7.15, 4.45, "re-entrainement", fontsize=10, color=SUBTEXT)

    # Legendes de sections
    ax.text(0.9, 7.35, "Pipeline MLOps", fontsize=10, color=SUBTEXT)
    ax.text(3.0, 3.65, "Boucle terrain", fontsize=10, color=SUBTEXT)

    return _to_png(fig)

def diagram_chargement_dataset() -> bytes:
    """Flux des données : Chargement du dataset intitial."""
    fig, ax = _make_canvas(16, 7)
    y = 5.3
    _soft_panel(ax, 0.4, 3.9, 11.5, 4.0)
    _box(ax, 0.6, y, 2.1, 1.2, "Kaggle", 13)
    _box(ax, 3.1, y, 2.8, 1.2, "Téléchargement\n+ structuration", 11)
    _box(ax, 8.6, y+1, 3.1, 1.2, "Supabase\nimages_dataset", 11)
    _box(ax, 8.6, y-1, 3.1, 1.2, "S3", 12)
    _arrow(ax, 2.7, 5.9, 3.1, 5.9)
    _arrow(ax, 5.9, 5.9, 8.6, y+1.5)
    _arrow(ax, 5.9, 5.9, 8.6, y-0.5)

    ax.text(6.7, y+1.2, "labels", fontsize=11, color=SUBTEXT)
    ax.text(6.7, y-0.3, "images", fontsize=11, color=SUBTEXT)

    ax.text(0.6, 8.8, "Flux des données", fontsize=18, weight="bold", color=TEXT)
    ax.text(0.6, 8.35, "Chargement du dataset intitial", fontsize=11, color=SUBTEXT)
    return _to_png(fig)

def diagram_flux_donnees() -> bytes:
    """Flux des données : version courte et lisible."""
    fig, ax = _make_canvas(16, 7)
    y = 5.3 
    _soft_panel(ax, 0.4, 4.95, 15.0, 2.0)
    _box(ax, 0.6, y, 2.1, 1.2, "Supabase\nimages_dataset", 13)
    _box(ax, 3.1, y, 2.8, 1.2, "Train/Test\n(0 & 1)", 12)
    _box(ax, 6.3, y, 2.5, 1.2, "Streaming (lots batch) S3\n+ structuration", 11)
    _box(ax, 9.2, y, 2.6, 1.2, "Oversampling", 12)
    _box(ax, 12.2, y, 2.9, 1.2, "Entraînement", 13)
    _arrow(ax, 2.7, 5.9, 3.1, 5.9)
    _arrow(ax, 5.9, 5.9, 6.3, 5.9)
    _arrow(ax, 8.8, 5.9, 9.2, 5.9)
    _arrow(ax, 11.8, 5.9, 12.2, 5.9)

    ax.text(0.6, 8.8, "Flux des données", fontsize=18, weight="bold", color=TEXT)
    ax.text(0.6, 8.35, "Préparation, enrichissement et réutilisation des données terrain", fontsize=11, color=SUBTEXT)
    return _to_png(fig)


def diagram_pipeline() -> bytes:
    """Pipeline d'entrainement avec logs MLflow."""

    fig, ax = _make_canvas(16, 10)
    y_top = 5.0     # pipeline d'entraînement
    y_bottom = 2.0  # pipeline MLflow

    # ------------------------
    # Pipeline principal d'entrainement
    # ------------------------
    _soft_panel(ax, 0.45, y_top-0.3, 10.1, 1.75)
    labels_train = ["load_data", "fit", "evaluate", "save"]
    x_train = [0.7, 3.2, 5.7, 8.2]
    for i, lb in enumerate(labels_train):
        _box(ax, x_train[i], y_top, 2.1, 1.2, f"{i+1}. {lb}", 12)
    for i in range(len(labels_train) - 1):
        _arrow(ax, x_train[i] + 2.1, y_top + 0.6, x_train[i + 1], y_top + 0.6)

    ax.text(0.7, y_top + 2.0, "Pipeline d'entraînement", fontsize=18, weight="bold", color=TEXT)
    ax.text(0.7, y_top + 1.6, "Exécution séquentielle automatisée en 4 étapes", fontsize=11, color=SUBTEXT)

    # ------------------------
    # Pipeline MLflow empilé en dessous
    # ------------------------
    _soft_panel(ax, 0.45, y_bottom - 0.3, 10.1, 1.75)
    labels_mlflow = ["log_params", "log_metrics", "log_artifacts", "register_model"]
    x_mlflow = [0.7, 3.2, 5.7, 8.2]
    for i, lb in enumerate(labels_mlflow):
        _box(ax, x_mlflow[i], y_bottom, 2.1, 1.2, f"{i+1}. {lb}", 12)

    ax.text(0.7, y_bottom -0.8, "Pipeline Logging & Model Tracking (MLflow)", fontsize=18, weight="bold", color=TEXT)
    ax.text(0.7, y_bottom -1.2, "Collecte en parallèle des paramètres, métriques et artefacts", fontsize=11, color=SUBTEXT)

    # ------------------------
    # Optionnel : flèches verticales pour montrer la correspondance
    # ------------------------
    for i in range(len(labels_train)):
        _arrow(ax, x_mlflow[i] + 1.05, y_top, x_train[i] + 1.05, y_bottom + 1.2)

    return _to_png(fig)


def diagram_endpoints() -> bytes:
    """Vue claire des endpoints : client vers endpoints."""
    fig, ax = _make_canvas(16, 8.5)
    _box(ax, 0.8, 4.2, 3.0, 1.4, "Client\n(App / script)", 12)
    _soft_panel(ax, 5.8, 1.35, 8.95, 6.95)

    endpoints = [
        "GET /health",
        "POST /train",
        "POST /predict",
        "POST /predict-with-gradcam",
        "POST /predict-from-db",
        "POST /feedback",
    ]
    ys = [7.2, 6.1, 5.0, 3.9, 2.8, 1.7]
    for ep, y in zip(endpoints, ys):
        _box(ax, 6.0, y, 8.6, 0.9, ep, 11)
        _arrow(ax, 3.8, 4.9, 6.0, y + 0.45)

    ax.text(0.8, 9.2, "Endpoints API", fontsize=18, weight="bold", color=TEXT)
    ax.text(0.8, 8.75, "Points d'entrée opérationnels pour train / predict / feedback", fontsize=11, color=SUBTEXT)
    return _to_png(fig)


def diagram_boucle_feedback() -> bytes:
    """Boucle de feedback en cycle (lisible)."""
    fig, ax = _make_canvas(16, 8)

    _soft_panel(ax, 0.7, 2.2, 14.6, 6.2)
    _box(ax, 5.6, 7.1, 4.8, 1.1, "1) Prédiction en usage terrain", 12)
    _box(ax, 10.3, 4.8, 4.8, 1.1, "2) Feedback médecin\n(/feedback)", 11)
    _box(ax, 5.6, 2.5, 4.8, 1.1, "3) Base Supabase mise à jour\n(images_dataset + feedback)", 11)
    _box(ax, 0.9, 4.8, 4.8, 1.1, "4) Ré-entraînement\n(data_source=db)", 11)

    _arrow(ax, 10.4, 7.1, 12.6, 5.9)
    _arrow(ax, 10.3, 4.8, 8.0, 3.6)
    _arrow(ax, 5.6, 3.0, 3.3, 4.8)
    _arrow(ax, 3.3, 5.9, 5.6, 7.1)

    ax.text(0.9, 9.2, "Boucle de feedback MLOps", fontsize=18, weight="bold", color=TEXT)
    ax.text(0.9, 8.75, "Apprentissage continu piloté par le diagnostic médecin", fontsize=11, color=SUBTEXT)
    return _to_png(fig)


__all__ = [
    "diagram_architecture",
    "diagram_flux_donnees",
    "diagram_pipeline",
    "diagram_endpoints",
    "diagram_boucle_feedback",
]
