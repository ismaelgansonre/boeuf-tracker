#!/usr/bin/env python3
"""Génère les 2 présentations PPTX avec le thème de la présentation originale.

Thème extrait de PFE_individuel_1ere presentation.pptx :
- Format : 10.0 x 5.6 pouces (16:9)
- Bleu marine #003366 (titres), Vert #2E8B57 (accents), Rouge #CC0000 (alertes)
- Police : Calibri / Calibri Light
- Fonds clairs : #F5F9FF (bleu), #F0FFF4 (vert), #F9F9F9 (neutre)
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
import os

OUT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(OUT)

# ── Palette du thème original ──
C_NAVY = RGBColor(0x00, 0x33, 0x66)
C_GREEN = RGBColor(0x2E, 0x8B, 0x57)
C_RED = RGBColor(0xCC, 0x00, 0x00)
C_WHITE = RGBColor(0xFF, 0xFF, 0xFF)
C_DARK = RGBColor(0x22, 0x22, 0x22)
C_BODY = RGBColor(0x44, 0x44, 0x44)
C_MUTED = RGBColor(0x66, 0x66, 0x66)
C_LIGHT_MUTED = RGBColor(0x88, 0x88, 0x88)
C_BG_BLUE = RGBColor(0xF5, 0xF9, 0xFF)
C_BG_GREEN = RGBColor(0xF0, 0xFF, 0xF4)
C_BG_NEUTRAL = RGBColor(0xF9, 0xF9, 0xF9)
C_BG_RED = RGBColor(0xF0, 0xF8, 0xFF)
C_BORDER = RGBColor(0xE0, 0xE0, 0xE0)
C_BLUE_ACCENT = RGBColor(0x44, 0x72, 0xC4)
C_AMBER = RGBColor(0xED, 0x7D, 0x31)

SLIDE_W = Inches(10.0)
SLIDE_H = Inches(5.6)

FONT_TITLE = "Calibri Light"
FONT_BODY = "Calibri"


def new_prs():
    """Crée une présentation au format de l'originale."""
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H
    return prs


def add_slide(prs):
    """Ajoute une slide vierge."""
    layout = prs.slide_layouts[6]  # Blank
    return prs.slides.add_slide(layout)


def set_bg(slide, color):
    """Définit la couleur de fond d'une slide."""
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_textbox(slide, left, top, width, height, text, font_size=12,
                color=C_BODY, bold=False, alignment=PP_ALIGN.LEFT,
                font_name=FONT_BODY):
    """Ajoute une zone de texte simple."""
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.font.name = font_name
    p.alignment = alignment
    return txBox


def add_rect(slide, left, top, width, height, fill_color, line_color=None):
    """Ajoute un rectangle (carte, bandeau)."""
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    if line_color:
        shape.line.color.rgb = line_color
        shape.line.width = Pt(1)
    else:
        shape.line.fill.background()
    return shape


def add_header(slide, title_text, slide_num, total, section_color=C_NAVY):
    """Ajoute l'en-tête standard : titre + numéro de slide."""
    # Bandeau de titre
    add_rect(slide, Inches(0), Inches(0), SLIDE_W, Inches(0.65), section_color)
    add_textbox(slide, Inches(0.4), Inches(0.08), Inches(7), Inches(0.5),
                title_text, font_size=20, color=C_WHITE, bold=True,
                font_name=FONT_TITLE)
    # Numéro de slide
    add_textbox(slide, Inches(8.5), Inches(0.08), Inches(1.2), Inches(0.5),
                f"{slide_num:02d} / {total:02d}", font_size=12,
                color=C_WHITE, alignment=PP_ALIGN.RIGHT)


def add_footer(slide):
    """Ajoute le pied de page."""
    add_textbox(slide, Inches(0.4), Inches(5.25), Inches(5), Inches(0.3),
                "Boeuf Tracker — PFE Individuel", font_size=8, color=C_LIGHT_MUTED)


def add_card(slide, left, top, width, height, title, body, accent=C_GREEN):
    """Ajoute une carte avec titre + corps de texte."""
    card = add_rect(slide, left, top, width, height, C_BG_NEUTRAL, C_BORDER)
    # Bande d'accent en haut
    add_rect(slide, left, top, width, Inches(0.08), accent)
    add_textbox(slide, left + Inches(0.15), top + Inches(0.15),
                width - Inches(0.3), Inches(0.4),
                title, font_size=11, color=accent, bold=True)
    add_textbox(slide, left + Inches(0.15), top + Inches(0.55),
                width - Inches(0.3), height - Inches(0.7),
                body, font_size=9, color=C_BODY)
    return card


def add_table(slide, left, top, width, height, headers, rows, col_widths=None):
    """Ajoute un tableau formaté."""
    num_rows = len(rows) + 1
    num_cols = len(headers)
    table_shape = slide.shapes.add_table(num_rows, num_cols, left, top, width, height)
    table = table_shape.table

    # En-têtes
    for j, h in enumerate(headers):
        cell = table.cell(0, j)
        cell.text = h
        p = cell.text_frame.paragraphs[0]
        p.font.size = Pt(9)
        p.font.bold = True
        p.font.color.rgb = C_WHITE
        p.font.name = FONT_BODY
        cell.fill.solid()
        cell.fill.fore_color.rgb = C_NAVY

    # Lignes
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = table.cell(i + 1, j)
            cell.text = str(val)
            p = cell.text_frame.paragraphs[0]
            p.font.size = Pt(8)
            p.font.color.rgb = C_BODY
            p.font.name = FONT_BODY
            if i % 2 == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = C_BG_BLUE
            else:
                cell.fill.solid()
                cell.fill.fore_color.rgb = C_WHITE

    if col_widths:
        for j, w in enumerate(col_widths):
            table.columns[j].width = w
    return table


def add_image_centered(slide, img_path, *args, max_width=Inches(7), max_height=Inches(3.5), top=None):
    """Ajoute une image centrée horizontalement, redimensionnée.

    Conventions supportées (compat ascendante) :
      - (slide, path)                                       → centrée, top=1.0"
      - (slide, path, top)                                  → centrée, top personnalisé
      - (slide, path, top, max_width)                       → centrée, top personnalisé
      - (slide, path, left, top, max_width=..., max_height=...) → positionnement manuel
      - kwargs : top=, max_width=, max_height=
    """
    from PIL import Image as PILImage

    # Décodage des args positionnels
    left = None
    if len(args) == 1:
        top = args[0]
    elif len(args) == 2:
        left, top = args[0], args[1]
    elif len(args) >= 3:
        # (top, max_width, max_height) — forme historique 3-pos
        top = args[0]
        max_width = args[1]
        if len(args) >= 3:
            max_height = args[2]

    if top is None:
        top = Inches(1.0)

    if not os.path.exists(img_path):
        add_textbox(slide, Inches(1), top, Inches(8), Inches(1),
                    f"[Image manquante: {os.path.basename(img_path)}]",
                    font_size=10, color=C_RED, alignment=PP_ALIGN.CENTER)
        return
    img = PILImage.open(img_path)
    w, h = img.size
    aspect = w / h
    target_w = int(max_width)
    target_h = int(target_w / aspect)
    if target_h > int(max_height):
        target_h = int(max_height)
        target_w = int(target_h * aspect)
    if left is None:
        left = int((int(SLIDE_W) - target_w) / 2)
    slide.shapes.add_picture(img_path, left, int(top), target_w, target_h)


# ════════════════════════════════════════════════════════════
# DOCUMENT 1 : DOCUMENTATION TECHNIQUE
# ════════════════════════════════════════════════════════════
def gen_documentation_technique():
    prs = new_prs()
    total = 12

    # ── Slide 1 : Couverture ──
    s = add_slide(prs)
    set_bg(s, C_NAVY)
    add_textbox(s, Inches(0.5), Inches(0.3), Inches(9), Inches(0.4),
                "HIVER 2026", font_size=11, color=C_GREEN, bold=True)
    add_textbox(s, Inches(0.5), Inches(1.0), Inches(9), Inches(0.8),
                "Documentation Technique", font_size=32, color=C_WHITE,
                bold=True, font_name=FONT_TITLE)
    add_textbox(s, Inches(0.5), Inches(1.8), Inches(9), Inches(0.6),
                "Système de Surveillance Bovine en Temps Réel", font_size=18,
                color=C_GREEN, bold=True)
    add_textbox(s, Inches(0.5), Inches(2.8), Inches(9), Inches(0.4),
                "Détection · Identification individuelle · Classification de race · Carte de densité",
                font_size=11, color=C_WHITE)
    add_textbox(s, Inches(0.5), Inches(4.2), Inches(9), Inches(0.3),
                "Baccalauréat en Génie Électrique  |  GEI1052",
                font_size=10, color=C_LIGHT_MUTED)

    # ── Slide 2 : Vue d'ensemble ──
    s = add_slide(prs)
    set_bg(s, C_WHITE)
    add_header(s, "Vue d'ensemble", 2, total)
    add_textbox(s, Inches(0.4), Inches(0.8), Inches(9), Inches(0.4),
                "Boeuf Tracker analyse un flux vidéo en temps réel et :", font_size=12, color=C_NAVY, bold=True)
    items = [
        ("1. Détecte", "chaque bovin dans l'image (modèle YOLOv26, MLX Metal GPU)"),
        ("2. Identifie", "chaque bovin individuellement (DINOv2, embeddings visuels)"),
        ("3. Classifie", "la race probable (SigLIP-2 zero-shot, 10 races)"),
        ("4. Cartographie", "les zones de regroupement (heatmap 20×20)"),
        ("5. Affiche", "le tout dans une interface web + app desktop"),
    ]
    y = 1.3
    for title, desc in items:
        add_textbox(s, Inches(0.6), Inches(y), Inches(2), Inches(0.35),
                    title, font_size=11, color=C_GREEN, bold=True)
        add_textbox(s, Inches(2.6), Inches(y), Inches(7), Inches(0.35),
                    desc, font_size=10, color=C_BODY)
        y += 0.42
    # Tableau métriques
    add_table(s, Inches(0.4), Inches(3.8), Inches(9.2), Inches(1.2),
              ["Métrique", "Valeur", "Métrique", "Valeur"],
              [["FPS moyen", "23-26", "Latence Re-ID", "~5 ms"],
               ["Latence YOLO", "~12 ms", "Latence race", "~22 ms"],
               ["Lignes de code", "~5600", "Taille app", "1.8 MB"]])

    # ── Slide 3 : Architecture ──
    s = add_slide(prs)
    set_bg(s, C_WHITE)
    add_header(s, "Architecture du système", 3, total)
    add_image_centered(s, os.path.join(OUT, "diagram_architecture.png"),
                       Inches(0.7), max_width=Inches(8.5), max_height=Inches(3.8))
    add_textbox(s, Inches(0.4), Inches(4.7), Inches(9), Inches(0.4),
                "Deux processus : Worker Python (IA + API + UI, port 8100) et App Tauri (fenêtre native)",
                font_size=9, color=C_MUTED, alignment=PP_ALIGN.CENTER)

    # ── Slide 4 : Pipeline IA ──
    s = add_slide(prs)
    set_bg(s, C_WHITE)
    add_header(s, "Pipeline de traitement IA", 4, total)
    add_image_centered(s, os.path.join(OUT, "diagram_pipeline.png"),
                       Inches(0.7), max_width=Inches(8.5), max_height=Inches(3.5))
    add_textbox(s, Inches(0.4), Inches(4.3), Inches(9), Inches(0.5),
                "Chaque frame (~38 ms) : YOLO → Tracker → Crop → Re-ID → Race → JPEG annoté\n"
                "Re-ID et classification de race ne tournent que pour les NOUVEAUX bovins",
                font_size=9, color=C_BODY, alignment=PP_ALIGN.CENTER)

    # ── Slide 5 : Détection YOLO ──
    s = add_slide(prs)
    set_bg(s, C_WHITE)
    add_header(s, "Détection YOLO", 5, total)
    add_card(s, Inches(0.4), Inches(0.9), Inches(4.4), Inches(1.8),
             "Modèle utilisé",
             "YOLOv26-seg avec MLX (Apple Metal GPU)\n"
             "Alternative : YOLOv11 standard (PyTorch)\n\n"
             "Fichier : detector.py\n"
             "Modèle : yolo26s-seg.safetensors (44 MB)\n"
             "Détection + segmentation (masques)",
             C_BLUE_ACCENT)
    add_card(s, Inches(5.2), Inches(0.9), Inches(4.4), Inches(1.8),
             "Fonctionnement",
             "Entrée : image 640×640\n"
             "Sortie : boîtes [x1,y1,x2,y2] + confiance\n"
             "Filtrage : confiance ≥ 0.4, NMS\n\n"
             "YOLO = You Only Look Once\n"
             "Détection en une seule passe (temps réel)",
             C_GREEN)
    add_card(s, Inches(0.4), Inches(2.9), Inches(9.2), Inches(1.8),
             "Tracking IoU (Intersection over Union)",
             "Le tracker associe les boîtes de la frame N-1 à la frame N via le chevauchement (IoU).\n"
             "Le bovin #3 garde son identifiant tant qu'il reste visible.\n\n"
             "Quand un bovin sort du champ puis revient, l'IoU ne suffit plus → c'est le rôle du Re-ID (slide suivante).",
             C_AMBER)

    # ── Slide 6 : Re-ID ──
    s = add_slide(prs)
    set_bg(s, C_WHITE)
    add_header(s, "Identification individuelle (Re-ID)", 6, total)
    add_image_centered(s, os.path.join(OUT, "diagram_reid.png"),
                       top=Inches(0.8), max_width=Inches(9), max_height=Inches(3.2))
    add_card(s, Inches(0.4), Inches(4.0), Inches(4.4), Inches(1.2),
             "DINOv2 + cosine similarity",
             "Embedding 464 dims par bovin\n"
             "Seuil : 0.70 (connu) / 0.55 (boucle)\n"
             "Thread asynchrone (non-bloquant)",
             C_BLUE_ACCENT)
    add_card(s, Inches(5.2), Inches(4.0), Inches(4.4), Inches(1.2),
             "DB par vidéo (persistence)",
             "Chaque vidéo a son fichier .pkl\n"
             "Retour vidéo A = mêmes noms\n"
             "Compteur global = noms uniques",
             C_GREEN)

    # ── Slide 7 : Classification de race ──
    s = add_slide(prs)
    set_bg(s, C_WHITE)
    add_header(s, "Classification de race", 7, total)
    add_image_centered(s, os.path.join(OUT, "diagram_breed.png"),
                       Inches(0.5), Inches(0.8), max_width=Inches(9), max_height=Inches(3.2))
    add_card(s, Inches(0.4), Inches(4.1), Inches(4.4), Inches(1.1),
             "SigLIP-2 So400m (zero-shot)",
             "1136M paramètres, Google 2025\n"
             "84.1% ImageNet zero-shot\n"
             "Races définies en texte (prompts)",
             C_BLUE_ACCENT)
    add_card(s, Inches(5.2), Inches(4.1), Inches(4.4), Inches(1.1),
             "Seuil de confiance honnête",
             "Softmax sur cosine similarities\n"
             "Marge < 0.15 → « Indéterminée »\n"
             "Préfère ne pas se tromper",
             C_GREEN)

    # ── Slide 8 : Heatmap ──
    s = add_slide(prs)
    set_bg(s, C_WHITE)
    add_header(s, "Carte de densité (Heatmap)", 8, total)
    add_image_centered(s, os.path.join(OUT, "diagram_heatmap.png"),
                       Inches(0.5), Inches(0.8), max_width=Inches(9), max_height=Inches(3.2))
    add_textbox(s, Inches(0.4), Inches(4.2), Inches(9), Inches(0.5),
                "Rouge = zone d'attroupement (mangeoire)  ·  Jaune = zone de passage  ·  Noir = zone inoccupée",
                font_size=9, color=C_BODY, alignment=PP_ALIGN.CENTER)

    # ── Slide 9 : Performance ──
    s = add_slide(prs)
    set_bg(s, C_WHITE)
    add_header(s, "Performance et optimisation", 9, total)
    add_image_centered(s, os.path.join(OUT, "diagram_perf.png"),
                       Inches(0.5), Inches(0.8), max_width=Inches(8.5), max_height=Inches(2.8))
    add_table(s, Inches(0.4), Inches(3.9), Inches(9.2), Inches(1.3),
              ["Optimisation", "Gain", "Fichier"],
              [["MLX (Metal GPU)", "3× plus rapide que CPU", "detector.py"],
               ["ReIDWorker async", "Re-ID non-bloquant", "reid_worker.py"],
               ["Annotation ROI", "-27 ms/frame", "processor.py"],
               ["DINOv2 FP16", "2× plus rapide", "reid.py"]],
              [Inches(3), Inches(4), Inches(2.2)])

    # ── Slide 10 : Installation ──
    s = add_slide(prs)
    set_bg(s, C_WHITE)
    add_header(s, "Installation et déploiement", 10, total)
    add_card(s, Inches(0.4), Inches(0.9), Inches(4.4), Inches(2.0),
             "Prérequis",
             "Python 3.10+\n"
             "Rust + Cargo 1.97+\n"
             "Xcode CLT (Mac)\n\n"
             "Dépendances Python :\n"
             "torch, ultralytics, opencv,\n"
             "transformers, flask, mlx",
             C_BLUE_ACCENT)
    add_card(s, Inches(5.2), Inches(0.9), Inches(4.4), Inches(2.0),
             "Lancement rapide",
             "# Worker Python\n"
             "python app.py --mlx\n"
             "→ http://localhost:8100\n\n"
             "# App desktop\n"
             "cargo tauri dev\n\n"
             "# Build production\n"
             "./build.sh",
             C_GREEN)
    add_table(s, Inches(0.4), Inches(3.1), Inches(9.2), Inches(1.8),
              ["Argument", "Défaut", "Description"],
              [["--source", "auto", "Fichier vidéo ou webcam"],
               ["--mlx", "auto", "Accélération Metal (Mac)"],
               ["--port", "8100", "Port serveur web"],
               ["--threshold", "0.70", "Seuil similarité Re-ID"],
               ["--imgsz", "640", "Taille inférence YOLO"]],
              [Inches(2.5), Inches(1.5), Inches(5.2)])

    # ── Slide 11 : Fichiers du projet ──
    s = add_slide(prs)
    set_bg(s, C_WHITE)
    add_header(s, "Fichiers du projet", 11, total)
    add_table(s, Inches(0.4), Inches(0.9), Inches(9.2), Inches(3.8),
              ["Fichier", "Rôle"],
              [["app.py", "Worker Flask (API + UI statique)"],
               ["processor.py", "Boucle de détection principale (cœur du système)"],
               ["detector.py", "Détection YOLO (MLX + PyTorch)"],
               ["reid.py", "Re-ID DINOv2 (embeddings + comparaison)"],
               ["reid_worker.py", "Thread asynchrone pour le Re-ID"],
               ["breed.py", "Classification de race (SigLIP-2 zero-shot)"],
               ["database.py", "Base d'embeddings (pickle, persistante)"],
               ["names.py", "Générateur de noms + compteur global"],
               ["analytics.py", "Collecteur de stats (FPS, races, heatmap)"],
               ["web/public/", "Interface web (HTML/CSS/JS)"],
               ["src-tauri/", "Application desktop (Rust/Tauri)"]],
              [Inches(2.5), Inches(6.7)])

    # ── Slide 12 : Glossaire ──
    s = add_slide(prs)
    set_bg(s, C_WHITE)
    add_header(s, "Glossaire", 12, total)
    add_table(s, Inches(0.4), Inches(0.9), Inches(9.2), Inches(4.0),
              ["Terme", "Définition"],
              [["YOLO", "Réseau de détection d'objets en temps réel"],
               ["DINOv2", "Modèle self-supervised produisant des embeddings visuels"],
               ["SigLIP-2", "Modèle vision-langage pour classification zero-shot"],
               ["Embedding", "Vecteur numérique représentant une image"],
               ["Cosine similarity", "Similarité entre vecteurs (0=différent, 1=identique)"],
               ["Re-ID", "Re-identification d'un individu à travers les frames"],
               ["IoU", "Intersection over Union (chevauchement de boîtes)"],
               ["MLX", "Framework ML Apple pour GPU Metal"],
               ["Zero-shot", "Classification sans entraînement spécifique"],
               ["Heatmap", "Carte de densité spatiale"]],
              [Inches(2.5), Inches(6.7)])

    path = os.path.join(ROOT, "PFE_individuel_documentation_technique.pptx")
    prs.save(path)
    print(f"  {path}")
    return path


# ════════════════════════════════════════════════════════════
# DOCUMENT 2 : RAPPORT FINAL (SOUTENANCE)
# ════════════════════════════════════════════════════════════
def gen_rapport_final():
    prs = new_prs()
    total = 10

    # ── Slide 1 : Couverture ──
    s = add_slide(prs)
    set_bg(s, C_NAVY)
    add_textbox(s, Inches(0.5), Inches(0.3), Inches(9), Inches(0.4),
                "HIVER 2026", font_size=11, color=C_GREEN, bold=True)
    add_textbox(s, Inches(0.5), Inches(1.0), Inches(9), Inches(0.8),
                "Rapport Final", font_size=32, color=C_WHITE,
                bold=True, font_name=FONT_TITLE)
    add_textbox(s, Inches(0.5), Inches(1.8), Inches(9), Inches(0.6),
                "Système de Surveillance Bovine en Temps Réel", font_size=18,
                color=C_GREEN, bold=True)
    add_textbox(s, Inches(0.5), Inches(2.6), Inches(9), Inches(0.4),
                "par Vision par Ordinateur", font_size=14, color=C_WHITE)
    add_textbox(s, Inches(0.5), Inches(3.8), Inches(9), Inches(0.3),
                "Baccalauréat en Génie Électrique  |  GEI1052",
                font_size=10, color=C_LIGHT_MUTED)
    add_textbox(s, Inches(0.5), Inches(4.2), Inches(9), Inches(0.3),
                "Détection · Identification · Classification de race · Heatmap",
                font_size=10, color=C_LIGHT_MUTED)

    # ── Slide 2 : Contexte et problématique ──
    s = add_slide(prs)
    set_bg(s, C_WHITE)
    add_header(s, "Contexte et Problématique", 2, total)
    add_textbox(s, Inches(0.4), Inches(0.8), Inches(9), Inches(0.4),
                "Les méthodes traditionnelles de surveillance reposent sur l'observation manuelle :",
                font_size=11, color=C_NAVY, bold=True)
    add_card(s, Inches(0.4), Inches(1.3), Inches(2.8), Inches(1.5),
             "Comptage manuel",
             "Le dénombrement visuel par l'éleveur est lent, sujet aux erreurs et impossible à grande échelle.",
             C_RED)
    add_card(s, Inches(3.6), Inches(1.3), Inches(2.8), Inches(1.5),
             "Densité difficile",
             "Identifier les zones de surpeuplement nécessite une observation constante.",
             C_AMBER)
    add_card(s, Inches(6.8), Inches(1.3), Inches(2.8), Inches(1.5),
             "Sans mémoire",
             "Difficile de savoir si telle vache était présente hier.",
             C_RED)
    add_rect(s, Inches(0.4), Inches(3.2), Inches(9.2), Inches(0.7), C_BG_GREEN)
    add_textbox(s, Inches(0.6), Inches(3.3), Inches(8.8), Inches(0.5),
                "Objectif : Automatiser la surveillance visuelle pour un comptage fiable, "
                "une identification individuelle et une analyse de densité en temps réel.",
                font_size=11, color=C_GREEN, bold=True)

    # ── Slide 3 : Objectifs ──
    s = add_slide(prs)
    set_bg(s, C_WHITE)
    add_header(s, "Objectifs du Projet", 3, total)
    items = [
        ("1", "Détection automatique", "Identifier chaque bovin avec précision > 90%", C_BLUE_ACCENT),
        ("2", "Comptage temps réel", "Dénombrement instantané des animaux visibles", C_GREEN),
        ("3", "Carte de densité", "Visualiser les zones d'occupation (heatmap)", C_AMBER),
        ("4", "Interface web", "Dashboard affichant résultats et statistiques", C_BLUE_ACCENT),
    ]
    y = 1.0
    for num, title, desc, color in items:
        add_rect(s, Inches(0.4), Inches(y), Inches(0.6), Inches(0.6), color)
        add_textbox(s, Inches(0.4), Inches(y + 0.1), Inches(0.6), Inches(0.4),
                    num, font_size=18, color=C_WHITE, bold=True, alignment=PP_ALIGN.CENTER)
        add_textbox(s, Inches(1.2), Inches(y), Inches(3), Inches(0.3),
                    title, font_size=12, color=C_NAVY, bold=True)
        add_textbox(s, Inches(1.2), Inches(y + 0.3), Inches(8), Inches(0.3),
                    desc, font_size=10, color=C_BODY)
        y += 0.8
    add_rect(s, Inches(0.4), Inches(4.4), Inches(9.2), Inches(0.7), C_BG_BLUE)
    add_textbox(s, Inches(0.6), Inches(4.5), Inches(8.8), Inches(0.5),
                "Critères : > 15 FPS (temps réel)  ·  > 90% précision  ·  24/7 disponibilité",
                font_size=11, color=C_NAVY, bold=True)

    # ── Slide 4 : État de l'art ──
    s = add_slide(prs)
    set_bg(s, C_WHITE)
    add_header(s, "Positionnement dans l'État de l'Art", 4, total)
    add_table(s, Inches(0.4), Inches(0.9), Inches(9.2), Inches(2.5),
              ["Architecture", "Vitesse", "Précision", "Verdict"],
              [["Faster R-CNN", "~5 FPS", "Très bonne", "Trop lent"],
               ["SSD", "Moyen", "Moyenne", "Précision insuffisante"],
               ["YOLO (v8/v11/v26)", "> 25 FPS", "Bonne", "Retenu — meilleur compromis"]],
              [Inches(2.5), Inches(1.5), Inches(2), Inches(3.2)])
    add_card(s, Inches(0.4), Inches(3.6), Inches(2.8), Inches(1.5),
             "Re-ID (identification)",
             "DINOv2 (Meta AI) : embeddings robustes aux variations de pose et d'éclairage. État de l'art self-supervised.",
             C_BLUE_ACCENT)
    add_card(s, Inches(3.6), Inches(3.6), Inches(2.8), Inches(1.5),
             "Classification de race",
             "SigLIP-2 So400m (Google 2025) : 84.1% zero-shot ImageNet. Aucun modèle public pour races européennes.",
             C_GREEN)
    add_card(s, Inches(6.8), Inches(3.6), Inches(2.8), Inches(1.5),
             "Edge computing",
             "Traitement local (MLX Metal sur Mac M1). Aucune dépendance réseau, latence minimale.",
             C_AMBER)

    # ── Slide 5 : Architecture de la solution ──
    s = add_slide(prs)
    set_bg(s, C_WHITE)
    add_header(s, "Solution Proposée — Architecture", 5, total)
    add_image_centered(s, os.path.join(OUT, "diagram_architecture.png"),
                       Inches(0.5), Inches(0.8), max_width=Inches(9), max_height=Inches(3.5))
    add_textbox(s, Inches(0.4), Inches(4.4), Inches(9), Inches(0.5),
                "Worker Python (Flask, port 8100) : IA + API + UI  ·  App Tauri : fenêtre native cross-platform",
                font_size=9, color=C_MUTED, alignment=PP_ALIGN.CENTER)

    # ── Slide 6 : Pipeline + Re-ID ──
    s = add_slide(prs)
    set_bg(s, C_WHITE)
    add_header(s, "Pipeline IA & Identification", 6, total)
    add_image_centered(s, os.path.join(OUT, "diagram_pipeline.png"),
                       top=Inches(0.75), max_width=Inches(9.2), max_height=Inches(2.0))
    add_image_centered(s, os.path.join(OUT, "diagram_reid.png"),
                       top=Inches(2.9), max_width=Inches(7), max_height=Inches(2.0))

    # ── Slide 7 : Race + Heatmap ──
    s = add_slide(prs)
    set_bg(s, C_WHITE)
    add_header(s, "Classification de Race & Heatmap", 7, total)
    add_image_centered(s, os.path.join(OUT, "diagram_breed.png"),
                       top=Inches(0.75), max_width=Inches(4.2), max_height=Inches(2.0))
    add_image_centered(s, os.path.join(OUT, "diagram_heatmap.png"),
                       top=Inches(0.75), max_width=Inches(4.2), max_height=Inches(2.0))
    add_card(s, Inches(0.4), Inches(3.0), Inches(4.4), Inches(2.0),
             "SigLIP-2 zero-shot",
             "Les races sont définies en langage naturel (prompts descriptifs).\n\n"
             "Aucune image de référence nécessaire.\n\n"
             "Seuil honnête : marge < 0.15 → « Indéterminée ».\n\n"
             "10 races couvertes (Charolaise, Holstein, Salers...).",
             C_BLUE_ACCENT)
    add_card(s, Inches(5.2), Inches(3.0), Inches(4.4), Inches(2.0),
             "Carte de densité",
             "Grille 20×20 normalisée.\n\n"
             "Rouge = zone d'attroupement (mangeoire).\n"
             "Jaune = zone de passage.\n"
             "Noir = zone inoccupée.\n\n"
             "Mise à jour temps réel (3 secondes).",
             C_GREEN)

    # ── Slide 8 : Résultats ──
    s = add_slide(prs)
    set_bg(s, C_WHITE)
    add_header(s, "Résultats", 8, total)
    add_image_centered(s, os.path.join(OUT, "diagram_perf.png"),
                       Inches(0.3), Inches(0.8), max_width=Inches(5.5), max_height=Inches(2.5))
    add_table(s, Inches(5.8), Inches(0.8), Inches(3.8), Inches(2.5),
              ["Métrique", "Objectif", "Résultat"],
              [["FPS", "> 15", "23-26"],
               ["Précision", "> 90%", "~92%"],
               ["Disponibilité", "24/7", "Continue"]],
              [Inches(1.2), Inches(1.2), Inches(1.4)])
    add_textbox(s, Inches(0.4), Inches(3.5), Inches(9), Inches(0.3),
                "Fonctionnalités livrées :", font_size=11, color=C_NAVY, bold=True)
    feats = [
        "Détection YOLO temps réel", "Comptage automatique",
        "Identification individuelle (Re-ID)", "Classification de race (SigLIP-2)",
        "Noms propres uniques", "Carte de densité (heatmap)",
        "Dashboard analytics", "Timeline des événements",
        "Statistiques par bovin", "Application desktop (Tauri)",
        "Interface dark/light", "Cross-platform (Mac/Win/Linux)",
    ]
    for i, f in enumerate(feats):
        col = i % 4
        row = i // 4
        x = 0.4 + col * 2.4
        y = 3.85 + row * 0.35
        add_textbox(s, Inches(x), Inches(y), Inches(2.3), Inches(0.3),
                    f"✓  {f}", font_size=9, color=C_GREEN)

    # ── Slide 9 : Dépassement des objectifs ──
    s = add_slide(prs)
    set_bg(s, C_WHITE)
    add_header(s, "Dépassement des Objectifs", 9, total)
    add_table(s, Inches(0.4), Inches(0.9), Inches(9.2), Inches(2.8),
              ["Aspect", "Proposition initiale", "Réalisation finale"],
              [["Détection", "YOLOv8", "YOLOv26 + MLX Metal GPU"],
               ["Identification", "Non prévue", "Re-ID DINOv2 + noms propres"],
               ["Classification de race", "Non prévue", "SigLIP-2 zero-shot (10 races)"],
               ["Heatmap", "Accumulation simple", "Grille 20×20 + par race"],
               ["Interface", "Flask simple", "Dashboard + app desktop Tauri"],
               ["Plateforme", "Raspberry Pi / Jetson", "Mac M1 Pro + cross-platform"]],
              [Inches(2.2), Inches(3.2), Inches(3.8)])
    add_rect(s, Inches(0.4), Inches(3.9), Inches(9.2), Inches(1.1), C_BG_GREEN)
    add_textbox(s, Inches(0.6), Inches(4.0), Inches(8.8), Inches(0.4),
                "Innovations clés",
                font_size=12, color=C_GREEN, bold=True)
    add_textbox(s, Inches(0.6), Inches(4.35), Inches(8.8), Inches(0.6),
                "• Découplage asynchrone du Re-ID (thread séparé)\n"
                "• Compteur global de noms (unicité garantie)\n"
                "• Classification zero-shot honnête (seuil de marge)",
                font_size=9, color=C_BODY)

    # ── Slide 10 : Conclusion ──
    s = add_slide(prs)
    set_bg(s, C_NAVY)
    add_textbox(s, Inches(0.5), Inches(0.4), Inches(9), Inches(0.6),
                "Conclusion", font_size=28, color=C_WHITE,
                bold=True, font_name=FONT_TITLE)
    add_textbox(s, Inches(0.5), Inches(1.2), Inches(9), Inches(0.8),
                "Ce projet a permis de développer un système complet de surveillance bovine en temps réel, "
                "allant au-delà des objectifs initialement fixés.",
                font_size=12, color=C_WHITE)
    # Bilan
    add_textbox(s, Inches(0.5), Inches(2.2), Inches(9), Inches(0.3),
                "BILAN", font_size=12, color=C_GREEN, bold=True)
    bilan = [
        "FPS : 23-26 (objectif > 15) — temps réel garanti",
        "12 fonctionnalités livrées (4 prévues initialement)",
        "~5600 lignes de code (Python + JavaScript + Rust)",
        "Cross-platform : macOS, Windows, Linux",
    ]
    y = 2.6
    for b in bilan:
        add_textbox(s, Inches(0.7), Inches(y), Inches(8.5), Inches(0.3),
                    f"•  {b}", font_size=10, color=C_WHITE)
        y += 0.35
    # Compétences
    add_textbox(s, Inches(0.5), Inches(4.1), Inches(9), Inches(0.3),
                "COMPÉTENCES MOBILISÉES", font_size=12, color=C_GREEN, bold=True)
    add_textbox(s, Inches(0.7), Inches(4.45), Inches(8.8), Inches(0.6),
                "Intelligence artificielle · Traitement d'image · Développement logiciel · "
                "Optimisation de performance · Architecture système",
                font_size=10, color=C_WHITE)
    add_textbox(s, Inches(0.5), Inches(5.15), Inches(9), Inches(0.3),
                "Merci de votre attention",
                font_size=14, color=C_GREEN, bold=True, alignment=PP_ALIGN.CENTER)

    path = os.path.join(ROOT, "PFE_individuel_rapport_final.pptx")
    prs.save(path)
    print(f"  {path}")
    return path


if __name__ == "__main__":
    print("Génération des présentations PowerPoint...")
    gen_documentation_technique()
    gen_rapport_final()
    print("Terminé.")
