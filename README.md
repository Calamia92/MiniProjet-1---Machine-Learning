# Mini-projet Machine Learning

Ce projet contient trois experiences :

- PMC/MLP from scratch en NumPy sur XOR.
- PMC/MLP from scratch en NumPy sur une spirale 2D.
- Reseau Keras/TensorFlow sur MNIST.

## Installation

```bash
pip install -r requirements.txt
```

Pour la partie MNIST, TensorFlow/Keras doit aussi etre installe :

```bash
pip install -r requirements-mnist.txt
```

Note : l'environnement actuel du projet est en Python 3.14. Si `pip` ne trouve pas de version TensorFlow compatible, cree un environnement avec une version Python supportee par TensorFlow, puis relance l'installation.

## Lancer XOR et spirale

```bash
python main.py
```

Commandes separees :

```bash
python main.py xor
python main.py spiral
```

La spirale genere aussi `spiral_decision_boundary.png`.

## Lancer MNIST avec Keras

```bash
python mnist_keras.py
```

Alternative plus simple si TensorFlow ne s'installe pas localement : ouvrir `mnist_keras_colab.ipynb` dans Google Colab et executer les cellules.

## Structure

- `mlp_numpy.py` : implementation du PMC from scratch avec forward propagation, backpropagation, loss et prediction.
- `datasets.py` : generation de XOR et de la spirale 2D.
- `plotting.py` : affichage de la frontiere de decision.
- `main.py` : point d'entree pour XOR et spirale.
- `mnist_keras.py` : modele Keras pour MNIST.
