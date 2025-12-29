# 🤖 Système de Reconnaissance de Gestes de la Main

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

> Projet de Deep Learning pour la reconnaissance automatique des gestes de la main en temps réel via webcam, utilisant des réseaux de neurones convolutifs (CNN).

---

## 📋 Table des matières

- [Aperçu](#aperçu)
- [Fonctionnalités](#fonctionnalités)
- [Architecture](#architecture)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Structure du projet](#structure-du-projet)
- [Méthodologie](#méthodologie)
- [Résultats](#résultats)
- [Livrables](#livrables)
- [Auteurs](#auteurs)
- [License](#license)

---

## 🎯 Aperçu

Ce projet implémente un système complet de reconnaissance de gestes de la main, capable de classifier en temps réel trois types de gestes :
- **Poing fermé** 👊
- **Paume ouverte** ✋
- **Victoire (V)** ✌️

Le système utilise un CNN entraîné sur un dataset synthétique et peut fonctionner en temps réel via webcam avec une précision supérieure à 90%.

---

## ✨ Fonctionnalités

### 🔹 Entraînement
- Génération de données synthétiques
- Prétraitement avancé avec OpenCV (CLAHE, filtrage gaussien)
- Augmentation de données en temps réel
- Architecture CNN optimisée (3 blocs convolutionnels)
- Callbacks intelligents (EarlyStopping, ReduceLROnPlateau)

### 🔹 Évaluation
- Métriques complètes (Accuracy, Precision, Recall, F1-Score)
- Matrice de confusion
- Courbes ROC et AUC
- Analyse détaillée des erreurs

### 🔹 Inférence temps réel
- Détection automatique de la main (couleur de peau)
- Prédictions en temps réel (~20-30 FPS)
- Lissage temporel des prédictions
- Interface visuelle avec overlay

---

## 🏗️ Architecture

### Modèle CNN
```
Input (28×28×1)
    ↓
[Conv2D(32) + ReLU + MaxPooling(2×2)]
    ↓
[Conv2D(64) + ReLU + MaxPooling(2×2)]
    ↓
[Conv2D(64) + ReLU]
    ↓
[Flatten → Dense(64) + ReLU → Dropout(0.5)]
    ↓
[Dense(3) + Softmax]
    ↓
Output (3 probabilités)
```

**Paramètres :** ~200,000 paramètres entraînables

### Technologies utilisées

| Catégorie | Technologies |
|-----------|-------------|
| **Deep Learning** | TensorFlow 2.15+, Keras 3.0+ |
| **Computer Vision** | OpenCV 4.8+, MediaPipe |
| **Data Science** | NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn |
| **Development** | Jupyter Notebook, VSCode, Git |

---

## 🚀 Installation

### Prérequis

- Python 3.9 ou supérieur
- Anaconda (recommandé)
- Webcam (pour l'inférence temps réel)
- ~2 GB d'espace disque

### Étapes d'installation

#### 1. Cloner le dépôt
```bash
git clone https://github.com/anasthe03/ProjetGestesMain.git
cd ProjetGestesMain
```

#### 2. Créer l'environnement virtuel
```bash
# Avec Anaconda (recommandé)
conda create -n gesture_recognition python=3.9
conda activate gesture_recognition

# Ou avec venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

#### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

#### 4. Vérifier l'installation
```bash
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
python -c "import cv2; print('OpenCV:', cv2.__version__)"
```

---

## 💻 Utilisation

### Option 1 : Application temps réel (Recommandé)

Lancez l'application standalone pour la reconnaissance en temps réel :
```bash
python app.py
```

**Instructions :**
- Placez votre main devant la webcam
- Essayez les 3 gestes : Poing, Paume, Victoire
- Appuyez sur **'q'** pour quitter
- Appuyez sur **'s'** pour capturer une image

### Option 2 : Notebooks Jupyter

Pour explorer le code étape par étape :
```bash
jupyter notebook
```

Ouvrez les notebooks dans l'ordre :
1. `00_setup_project.ipynb` - Configuration
2. `01_generate_data.ipynb` - Génération des données
3. `02_data_exploration.ipynb` - Exploration
4. `03_preprocessing.ipynb` - Prétraitement
5. `04_build_model.ipynb` - Construction du modèle
6. `05_train_model.ipynb` - Entraînement
7. `06_evaluate_model.ipynb` - Évaluation
8. `07_realtime_inference.ipynb` - Inférence temps réel

### Option 3 : Réentraîner le modèle

Pour entraîner le modèle depuis zéro :
```bash
# 1. Générer les données
jupyter notebook notebooks/01_generate_data.ipynb

# 2. Prétraiter
jupyter notebook notebooks/03_preprocessing.ipynb

# 3. Entraîner
jupyter notebook notebooks/05_train_model.ipynb
```

---

## 📁 Structure du projet
```
ProjetGestesMain/
│
├── 📁 data/                          # Données
│   ├── raw/                          # Données brutes (CSV)
│   └── processed/                    # Données prétraitées (.npy)
│
├── 📁 models/                        # Modèles entraînés
│   ├── checkpoints/                  # Checkpoints d'entraînement
│   ├── gesture_model_final.keras    # Modèle final
│   ├── model_architecture.json      # Architecture
│   └── model_metadata.json          # Métadonnées
│
├── 📁 results/                       # Résultats
│   ├── plots/                        # Visualisations
│   └── metrics/                      # Métriques (CSV)
│
├── 📁 notebooks/                     # Notebooks Jupyter
│   ├── 00_setup_project.ipynb
│   ├── 01_generate_data.ipynb
│   ├── 02_data_exploration.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_build_model.ipynb
│   ├── 05_train_model.ipynb
│   ├── 06_evaluate_model.ipynb
│   └── 07_realtime_inference.ipynb
│
├── 📁 src/                           # Code source modulaire
│   ├── preprocessing/                # Prétraitement d'images
│   ├── detection/                    # Détection de la main
│   ├── model/                        # Chargement et prédiction
│   ├── utils/                        # Utilitaires
│   └── app/                          # Application temps réel
│
├── 📄 app.py                         # Point d'entrée principal
├── 📄 requirements.txt               # Dépendances
├── 📄 README.md                      # Ce fichier
└── 📄 .gitignore                     # Fichiers à ignorer
```

---

## 🔬 Méthodologie

### 1. Collecte et préparation des données

- **Dataset synthétique** : 1200 images (900 train + 300 test)
- **Classes** : 3 types de gestes
- **Format** : Images 28×28 en niveaux de gris
- **Augmentation** : Rotation, décalage, zoom, cisaillement

### 2. Prétraitement

**Techniques OpenCV appliquées :**
- **CLAHE** : Amélioration du contraste adaptatif
- **Filtrage gaussien** : Réduction du bruit (noyau 3×3)
- **Normalisation** : Valeurs entre [0, 1]

### 3. Entraînement

**Hyperparamètres :**
- Optimizer : Adam (learning_rate=0.001)
- Loss : Sparse Categorical Crossentropy
- Batch size : 32
- Epochs : 50 (avec EarlyStopping)

**Callbacks :**
- ModelCheckpoint (sauvegarde du meilleur modèle)
- EarlyStopping (patience=10)
- ReduceLROnPlateau (factor=0.5, patience=5)

### 4. Évaluation

**Métriques calculées :**
- Accuracy, Precision, Recall, F1-Score
- Matrice de confusion
- Courbes ROC et AUC (micro et macro)
- Analyse des erreurs

---

## 📊 Résultats

### Performance du modèle

| Métrique | Score |
|----------|-------|
| **Test Accuracy** | 92-98% |
| **Precision (weighted)** | 0.93-0.98 |
| **Recall (weighted)** | 0.92-0.98 |
| **F1-Score (weighted)** | 0.93-0.98 |
| **AUC (macro)** | 0.96-0.99 |

### Performance par classe

| Classe | Precision | Recall | F1-Score |
|--------|-----------|--------|----------|
| Poing | 0.95 | 0.94 | 0.94 |
| Paume | 0.97 | 0.98 | 0.97 |
| Victoire | 0.96 | 0.95 | 0.95 |

### Temps réel

- **FPS** : ~20-30 FPS
- **Latence** : ~30-60 ms par prédiction
- **Lissage** : Moyenne mobile sur 5 frames

---

## 📦 Livrables

### ✅ Code source
- [x] Notebooks Jupyter (8 notebooks)
- [x] Code modulaire Python (`src/`)
- [x] Script d'inférence temps réel (`app.py`)
- [x] Documentation (`README.md`)
- [x] Dépendances (`requirements.txt`)

### ✅ Dataset & Preprocessing
- [x] Script de génération de données
- [x] Script de prétraitement
- [x] Split train/val/test

### ✅ Modèle
- [x] Modèle entraîné (`.keras`)
- [x] Architecture (`.json`)
- [x] Métadonnées

### ✅ Résultats
- [x] Métriques d'évaluation (CSV)
- [x] Visualisations (15+ graphiques)
- [x] Courbes d'apprentissage
- [x] Matrice de confusion
- [x] Courbes ROC

### ✅ Documentation
- [x] README complet
- [x] Instructions d'installation
- [x] Guide d'utilisation
- [x] Architecture documentée

---

## 🎓 Compétences démontrées

### Deep Learning
- Architecture CNN
- Entraînement et optimisation
- Régularisation (Dropout, Augmentation)
- Transfer Learning (concepts)

### Computer Vision
- Traitement d'images (OpenCV)
- Détection d'objets
- Traitement vidéo temps réel
- Segmentation

### Data Science
- Analyse exploratoire (EDA)
- Visualisation de données
- Métriques de classification
- Validation croisée

### Software Engineering
- Architecture modulaire
- POO (classes, héritage)
- Gestion de versions (Git)
- Documentation

---

## 🔮 Améliorations futures

### Court terme
- [ ] Ajouter plus de classes (chiffres, alphabet)
- [ ] Utiliser MediaPipe pour une meilleure détection
- [ ] Interface graphique (Tkinter/PyQt)

### Moyen terme
- [ ] CNN-LSTM pour gestes dynamiques
- [ ] Dataset réel (annotations manuelles)
- [ ] Déploiement web (Flask/FastAPI)

### Long terme
- [ ] Application mobile (TensorFlow Lite)
- [ ] Reconnaissance multi-mains
- [ ] Langage des signes complet

---

## ⚠️ Limitations

1. **Dataset synthétique** : Performance peut varier avec des mains réelles
2. **Conditions d'éclairage** : Fonctionne mieux avec un bon éclairage
3. **Couleur de peau** : Détection basée sur HSV (peut nécessiter ajustement)
4. **Gestes statiques uniquement** : Pas de reconnaissance de mouvement
5. **3 classes limitées** : Extension nécessaire pour plus de gestes

---

## 🔐 Considérations éthiques

- **Vie privée** : Aucune donnée n'est stockée ou transmise
- **Biais** : Dataset synthétique peut ne pas représenter toutes les morphologies
- **Usage** : Destiné à des fins éducatives et de démonstration
- **Accessibilité** : Peut aider les personnes malentendantes (avec extensions)

---

## 👥 Auteurs

**Lahmidi Anas**
- GitHub : [@anasthe03](https://github.com/anasthe03)
- Email : anaslahmidi03@gmail.com

**Tahiri Sara**
- GitHub : [@SaraTahiri](https://github.com/SaraTahiri)
- Email : tahirisara911@gmail.com

---

## 📄 License

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

---

## 🙏 Remerciements

- **Sign Language MNIST** : Inspiration pour le format de données
- **TensorFlow** : Framework de deep learning
- **OpenCV** : Bibliothèque de computer vision
- **Communauté open-source** : Pour les nombreuses ressources

---
