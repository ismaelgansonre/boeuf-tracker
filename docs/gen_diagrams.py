#!/usr/bin/env python3
"""Génère les diagrammes (PNG) pour la documentation Boeuf Tracker."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

OUT = os.path.dirname(os.path.abspath(__file__))

# Palette
C_BG = "#0f1115"
C_CARD = "#1a1f2e"
C_ACCENT = "#16a34a"
C_AMBER = "#d97706"
C_ROSE = "#e11d48"
C_BLUE = "#3b82f6"
C_PURPLE = "#8b5cf6"
C_CYAN = "#06b6d4"
C_WHITE = "#e8eaed"
C_MUTED = "#9ca3af"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def box(ax, x, y, w, h, text, color=C_CARD, textcolor=C_WHITE, fontsize=10, bold=False):
    """Dessine une boîte arrondie avec du texte."""
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                           linewidth=1.5, edgecolor=color, facecolor=color, alpha=0.9)
    ax.add_patch(rect)
    weight = "bold" if bold else "normal"
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            color=textcolor, fontsize=fontsize, fontweight=weight, wrap=True)


def arrow(ax, x1, y1, x2, y2, color=C_MUTED, label="", style="->"):
    """Dessine une flèche entre deux points."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=1.8))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx+0.15, my, label, color=color, fontsize=8, fontstyle="italic")


# ──────────────────────────────────────────────────────
# 1. Architecture globale
# ──────────────────────────────────────────────────────
def diagram_architecture():
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("Architecture du système Boeuf Tracker", fontsize=14, fontweight="bold", pad=15)

    # App desktop (Tauri)
    box(ax, 0.3, 5.2, 5, 1.4, "Application Desktop (Tauri)\nFenêtre native WKWebView", C_PURPLE, fontsize=10, bold=True)
    box(ax, 0.5, 3.5, 2, 1.2, "Flux vidéo\nannoté (JPEG)", C_CARD, C_CYAN, fontsize=9)
    box(ax, 2.8, 3.5, 2.2, 1.2, "Dashboard\nHeatmap + Charts", C_CARD, C_CYAN, fontsize=9)

    arrow(ax, 2.8, 5.2, 1.5, 4.7, C_PURPLE)
    arrow(ax, 2.8, 5.2, 3.9, 4.7, C_PURPLE)

    # HTTP
    arrow(ax, 5.3, 5.9, 6.5, 5.9, C_ACCENT, "HTTP\nlocalhost:8100", "->")

    # Worker Python
    box(ax, 6.5, 4.8, 5.2, 2.0, "Worker Python (Flask, port 8100)\nIA + API REST + UI statique", C_ACCENT, fontsize=10, bold=True)

    # Pipeline IA
    box(ax, 6.7, 3.2, 1.4, 1.2, "YOLO\nDétection", C_BLUE, fontsize=9)
    box(ax, 8.3, 3.2, 1.4, 1.2, "Tracker\nIoU", C_BLUE, fontsize=9)
    box(ax, 9.9, 3.2, 1.6, 1.2, "DINOv2\nRe-ID", C_AMBER, fontsize=9)

    arrow(ax, 7.4, 4.8, 7.4, 4.4, C_MUTED)
    arrow(ax, 8.1, 3.8, 8.3, 3.8, C_MUTED)
    arrow(ax, 9.7, 3.8, 9.9, 3.8, C_MUTED)

    # Composants
    box(ax, 6.7, 1.5, 1.7, 1.2, "SigLIP-2\nClassification\nrace", C_ROSE, fontsize=8)
    box(ax, 8.6, 1.5, 1.7, 1.2, "Analytics\nHeatmap +\nStats", C_CYAN, fontsize=8)
    box(ax, 10.5, 1.5, 1.2, 1.2, "DB\nPickle", C_MUTED, fontsize=9)

    arrow(ax, 9.0, 3.2, 7.5, 2.7, C_MUTED)
    arrow(ax, 9.0, 3.2, 9.4, 2.7, C_MUTED)
    arrow(ax, 9.0, 3.2, 11.0, 2.7, C_MUTED)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "diagram_architecture.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  diagram_architecture.png")


# ──────────────────────────────────────────────────────
# 2. Pipeline IA
# ──────────────────────────────────────────────────────
def diagram_pipeline():
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("Pipeline de traitement (chaque frame vidéo)", fontsize=14, fontweight="bold", pad=15)

    steps = [
        (0.3, "Frame\nvidéo\nbrute", C_MUTED, "1280×720\nBGR"),
        (2.0, "YOLO\nDétection", C_BLUE, "~12 ms\nMLX Metal"),
        (3.7, "Tracker\nIoU", C_BLUE, "~1 ms\nassocie ID"),
        (5.4, "Crop", C_CYAN, "~0.1 ms\nextrait zone"),
        (7.1, "Re-ID\nDINOv2", C_AMBER, "~5 ms\n(async)"),
        (8.8, "Race\nSigLIP-2", C_ROSE, "~22 ms\n(nouveau)"),
        (10.5, "JPEG\nannoté", C_ACCENT, "~3 ms"),
    ]
    for x, label, color, sub in steps:
        box(ax, x, 2.8, 1.3, 1.3, label, color, fontsize=9, bold=True)
        ax.text(x+0.65, 2.3, sub, ha="center", va="top", color=C_MUTED, fontsize=7.5)

    for i in range(len(steps)-1):
        x1 = steps[i][0] + 1.3
        x2 = steps[i+1][0]
        arrow(ax, x1, 3.45, x2, 3.45, C_ACCENT)

    # Note
    ax.text(6, 0.8, "Total : ~38 ms par frame (≈ 26 FPS)  |  Re-ID et Race ne tournent que pour les NOUVEAUX bovins",
            ha="center", fontsize=9, color=C_ACCENT, fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0fdf4", edgecolor=C_ACCENT))

    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "diagram_pipeline.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  diagram_pipeline.png")


# ──────────────────────────────────────────────────────
# 3. Re-ID (embeddings)
# ──────────────────────────────────────────────────────
def diagram_reid():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("Identification individuelle (Re-ID) par embeddings DINOv2",
                 fontsize=13, fontweight="bold", pad=15)

    # Crop entrée
    box(ax, 0.3, 3.5, 1.8, 1.5, "Crop du\nbovin\n(200×300 px)", C_CYAN, fontsize=9, bold=True)

    # DINOv2
    box(ax, 2.6, 3.5, 1.8, 1.5, "DINOv2-small\nMeta AI\n(self-supervised)", C_BLUE, fontsize=9, bold=True)
    arrow(ax, 2.1, 4.25, 2.6, 4.25, C_ACCENT)

    # Embedding
    box(ax, 4.9, 3.5, 1.8, 1.5, "Embedding\n464 dimensions\n(vecteur)", C_AMBER, fontsize=9, bold=True)
    arrow(ax, 4.4, 4.25, 4.9, 4.25, C_ACCENT)

    # Comparaison
    box(ax, 7.2, 3.5, 2.5, 1.5, "Cosine similarity\nvs DB embeddings", C_PURPLE, fontsize=9, bold=True)
    arrow(ax, 6.7, 4.25, 7.2, 4.25, C_ACCENT)

    # Branches
    box(ax, 6.0, 0.5, 3, 1.2, "sim ≥ 0.70 → BOVIN CONNU\n(nom existant conservé)", C_ACCENT, fontsize=9, bold=True)
    box(ax, 1.0, 0.5, 3.5, 1.2, "sim < 0.70 → NOUVEAU BOVIN\n(nouveau nom créé)", C_ROSE, fontsize=9, bold=True)

    arrow(ax, 7.5, 3.5, 7.5, 1.7, C_ACCENT)
    arrow(ax, 8.5, 3.5, 8.5, 1.7, C_ACCENT)

    # Seuil
    ax.text(5, 2.2, "Seuil de similarité : 0.70",
            ha="center", fontsize=10, color=C_AMBER, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fffbeb", edgecolor=C_AMBER))

    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "diagram_reid.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  diagram_reid.png")


# ──────────────────────────────────────────────────────
# 4. Heatmap
# ──────────────────────────────────────────────────────
def diagram_heatmap():
    import numpy as np
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Gauche : frame vidéo simulée
    ax1 = axes[0]
    ax1.set_title("Flux vidéo + positions détectées", fontsize=11, fontweight="bold")
    # Simuler une image
    img = np.ones((100, 160, 3)) * 0.3
    img[20:40, 20:40] = 0.6  # zone herbe
    img[60:80, 100:130] = 0.5
    ax1.imshow(img, extent=[0, 160, 0, 100], aspect="auto")
    # Bovins
    for (x, y) in [(30, 60), (35, 55), (90, 30), (95, 35), (100, 30), (120, 70), (45, 65)]:
        rect = mpatches.Rectangle((x-5, y-5), 10, 10, linewidth=2, edgecolor=C_ACCENT, facecolor="none")
        ax1.add_patch(rect)
        ax1.plot(x, y, "o", color=C_ROSE, markersize=4)
    ax1.set_xlabel("Largeur (px)")
    ax1.set_ylabel("Hauteur (px)")

    # Droite : heatmap
    ax2 = axes[1]
    ax2.set_title("Carte de densité (grille 20×20)", fontsize=11, fontweight="bold")
    # Simuler grille
    grid = np.zeros((20, 20))
    grid[12:15, 5:8] = [3, 5, 4]   # attroupement
    grid[10:13, 6:9] += [2, 4, 3]
    grid[14:17, 12:15] = [2, 3, 2]  # passage
    grid[5:7, 14:16] = [1, 1]
    # Lissage manuel (convolution simple, evite la dependance scipy)
    try:
        from scipy.ndimage import gaussian_filter
        grid = gaussian_filter(grid, sigma=1.2)
    except ImportError:
        # Fallback : moyenne 3x3
        g = np.zeros_like(grid)
        for i in range(20):
            for j in range(20):
                vals = []
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        ni, nj = i+di, j+dj
                        if 0 <= ni < 20 and 0 <= nj < 20:
                            vals.append(grid[ni, nj])
                g[i, j] = np.mean(vals)
        grid = g
    im = ax2.imshow(grid, cmap="RdYlGn_r", interpolation="bilinear", origin="lower")
    ax2.set_xlabel("Position X normalisée")
    ax2.set_ylabel("Position Y normalisée")

    # Légende
    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("Densité de présence", fontsize=9)

    fig.suptitle("Principe de la carte de densité (Heatmap)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "diagram_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  diagram_heatmap.png")


# ──────────────────────────────────────────────────────
# 5. Performance (répartition temps)
# ──────────────────────────────────────────────────────
def diagram_perf():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_title("Répartition du temps de traitement par frame", fontsize=13, fontweight="bold", pad=15)

    labels = ["Détection\nYOLO (MLX)", "Tracker\nIoU", "Crop", "Re-ID\n(async)", "Annotation\n+ JPEG", "Flask"]
    times = [12, 1, 0.1, 0, 3, 3]
    colors = [C_BLUE, C_BLUE, C_CYAN, C_ACCENT, C_AMBER, C_PURPLE]

    bars = ax.barh(labels, times, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Temps (ms)", fontsize=11)
    ax.set_xlim(0, 15)

    for bar, t in zip(bars, times):
        if t > 0:
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                    f"{t} ms", va="center", fontsize=9, color=C_MUTED)
        else:
            ax.text(0.3, bar.get_y() + bar.get_height()/2,
                    "0 ms (non-bloquant)", va="center", fontsize=8, color=C_ACCENT, fontstyle="italic")

    ax.axvline(x=42, color=C_ROSE, linestyle="--", linewidth=1.5, label="Seuil 24 FPS (42 ms)")
    ax.text(42.5, 4.5, "42 ms = 24 FPS", color=C_ROSE, fontsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "diagram_perf.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  diagram_perf.png")


# ──────────────────────────────────────────────────────
# 6. Classification de race (SigLIP-2)
# ──────────────────────────────────────────────────────
def diagram_breed():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("Classification de race par SigLIP-2 zero-shot",
                 fontsize=13, fontweight="bold", pad=15)

    # Textes (prompts)
    prompts = [
        "a photo of a Charolais cow, solid white cream coat",
        "a photo of a Holstein cow, black white spotted",
        "a photo of a Salers cow, dark mahogany red coat",
        "a photo of a Limousin cow, golden wheat fawn",
    ]
    for i, p in enumerate(prompts):
        y = 5.0 - i * 0.7
        box(ax, 0.3, y, 3.2, 0.55, p[:45], C_CARD, C_MUTED, fontsize=7.5)

    # SigLIP-2
    box(ax, 4.0, 2.0, 2.2, 1.8, "SigLIP-2\nSo400m\n(1136M params)", C_PURPLE, fontsize=9, bold=True)

    # Flèches texte -> SigLIP
    for i in range(4):
        y = 5.27 - i * 0.7
        arrow(ax, 3.5, y, 4.0, 3.3, C_MUTED)

    # Image
    box(ax, 0.3, 1.5, 3.2, 1.2, "Crop du bovin\n(image)", C_CYAN, fontsize=9, bold=True)
    arrow(ax, 3.5, 2.1, 4.0, 2.5, C_ACCENT)

    # Sortie
    box(ax, 6.8, 2.0, 2.8, 1.8, "Softmax →\nRace + confiance\nex: Normande 63%", C_ACCENT, fontsize=9, bold=True)
    arrow(ax, 6.2, 2.9, 6.8, 2.9, C_ACCENT)

    # Note seuil
    ax.text(5, 0.5, "Si marge < 0.15 → « Indéterminée » (approche honnête)",
            ha="center", fontsize=9, color=C_AMBER, fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fffbeb", edgecolor=C_AMBER))

    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "diagram_breed.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  diagram_breed.png")


if __name__ == "__main__":
    print("Génération des diagrammes...")
    diagram_architecture()
    diagram_pipeline()
    diagram_reid()
    diagram_heatmap()
    diagram_perf()
    diagram_breed()
    print("Terminé. Fichiers dans:", OUT)
