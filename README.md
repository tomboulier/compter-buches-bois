# Compteur de Bûches avec SAM2

Ce projet est une application simple permettant de compter des bûches de bois sur une photo en utilisant l'intelligence artificielle (SAM2 - Segment Anything Model 2).

L'interface graphique utilise **Gradio**, ce qui la rend moderne et facile à utiliser dans le navigateur.

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

## Utilisation

Pour lancer l'application :

```bash
uv run app.py
```

Cela va lancer un serveur local. Ouvrez le lien affiché (généralement `http://127.0.0.1:7860`) dans votre navigateur.

### Fonctionnement de l'interface

1. **Chargement** : Glissez-déposez ou cliquez pour charger une image (ex: `buches.jpg`).
2. **Comptage** : Cliquez simplement sur une bûche sur l'image.
   - Le modèle IA (SAM2) va détecter les contours de la bûche.
   - Un masque vert se superpose sur la bûche.
   - Le compteur s'incrémente.
3. **Correction** :
   - **Annuler dernier** : Retire la dernière bûche comptée.
   - **Réinitialiser** : Efface tout et remet le compteur à zéro.

## Note technique

Le projet utilise le modèle `facebook/sam2.1-hiera-tiny`. Il est léger et configuré pour tourner sur CPU.
L'interface est propulsée par Gradio.

### Dépendances spécifiques
Ce projet nécessite des versions spécifiques pour éviter les conflits :
- `gradio >= 4.0.0`
- `pillow < 11.0`
- `sam2` installé depuis la source git.
