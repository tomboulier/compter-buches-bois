# Compteur de Bûches avec SAM2

Ce projet est une application simple permettant de compter des bûches de bois sur une photo en utilisant l'intelligence artificielle (SAM2 - Segment Anything Model 2).

L'interface graphique permet de cliquer sur les bûches pour les segmenter et les compter automatiquement.

## Prérequis

- **Python 3.10+**
- **uv** (Gestionnaire de paquets Python rapide)

Si vous n'avez pas `uv`, vous pouvez l'installer avec :
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Installation

1. Cloner le dépôt :
   ```bash
   git clone https://github.com/tomboulier/compter-buches-bois.git
   cd compter-buches-bois
   ```

2. Installer les dépendances (automatique avec `uv`) :
   ```bash
   uv sync
   ```
   Ou manuellement si vous n'utilisez pas `uv` (déconseillé pour ce projet configuré avec uv) :
   ```bash
   pip install opencv-python torch torchvision numpy pillow sam2
   ```

## Utilisation

Pour lancer l'application :

```bash
uv run python app.py
```

### Fonctionnement de l'interface

1. **Chargement** : L'image `buches.jpg` est chargée par défaut. Vous pouvez en charger une autre via le bouton "Charger Image".
2. **Comptage** : Cliquez simplement sur une bûche (clic gauche).
   - Le modèle IA (SAM2) va détecter les contours de la bûche.
   - Un masque vert se superpose sur la bûche.
   - Le compteur s'incrémente.
3. **Correction** :
   - **Annuler dernier** : Retire la dernière bûche comptée.
   - **Réinitialiser** : Efface tout et remet le compteur à zéro.

## Modèle

Le projet utilise le modèle `facebook/sam2.1-hiera-tiny`. Il est léger et configuré pour tourner sur CPU par défaut pour une compatibilité maximale.
Lors du premier lancement, le modèle sera téléchargé automatiquement par la bibliothèque `sam2`.

## Structure du projet

- `app.py` : Le code principal de l'interface graphique.
- `buches.jpg` : Image d'exemple.
- `legacy/` : Anciens scripts de tests et d'expérimentations.
