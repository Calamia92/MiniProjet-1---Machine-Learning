# Rapport - Mini-projet Machine Learning

## Objectif

Le projet compare deux manieres de construire un reseau de neurones :

- une implementation manuelle d'un perceptron multicouche avec NumPy ;
- une implementation haut niveau avec Keras/TensorFlow pour MNIST.

La partie NumPy sert a comprendre les mecanismes internes : propagation avant, calcul de l'erreur, retropropagation et mise a jour des poids.

## Partie 1 - PMC from scratch en NumPy

Le fichier `mlp_numpy.py` contient une classe `MLP`. Elle gere :

- l'initialisation des poids et des biais ;
- les activations `sigmoid`, `tanh`, `relu`, `softmax` et `linear` ;
- la propagation avant ;
- la retropropagation ;
- la descente de gradient par batch ;
- le calcul de la loss ;
- la prediction.

Pour les problemes binaires XOR et spirale, la couche de sortie utilise une activation sigmoide et une loss de type entropie croisee binaire.

## Partie 2 - XOR

XOR est un probleme non lineaire. Un modele lineaire seul ne peut pas le resoudre, donc il faut au moins une couche cachee.

Architecture utilisee :

- entree : 2 neurones ;
- couche cachee : 4 neurones avec `tanh` ;
- sortie : 1 neurone avec `sigmoid`.

Resultat obtenu :

```text
accuracy = 1.000
```

Le reseau apprend correctement les quatre combinaisons de XOR.

## Partie 3 - Spirale 2D

La spirale 2D est un probleme de classification binaire plus complexe. Les classes sont melangees dans l'espace, donc la frontiere de decision doit etre non lineaire.

Architecture utilisee :

- entree : 2 neurones ;
- couche cachee 1 : 32 neurones avec `tanh` ;
- couche cachee 2 : 32 neurones avec `tanh` ;
- sortie : 1 neurone avec `sigmoid`.

Resultat obtenu :

```text
accuracy = 0.998
```

Le script genere aussi une visualisation de la frontiere de decision dans `spiral_decision_boundary.png`.

## Partie 4 - MNIST avec Keras

Le fichier `mnist_keras.py` contient un modele Keras pour classifier les chiffres manuscrits MNIST.

Architecture :

- `Flatten` pour transformer les images 28x28 en vecteurs ;
- `Dense(128, activation="relu")` ;
- `Dropout(0.2)` ;
- `Dense(10, activation="softmax")`.

La compilation utilise :

- optimiseur : `adam` ;
- loss : `sparse_categorical_crossentropy` ;
- metrique : `accuracy`.

Dans l'environnement actuel, Python est en version 3.14. TensorFlow n'est pas disponible pour cette version via `pip`, donc le script Keras est pret mais n'a pas pu etre execute localement. Pour executer cette partie, il faut creer un environnement Python compatible TensorFlow.

## Conclusion

La partie NumPy montre le fonctionnement interne d'un reseau de neurones. Elle valide que le PMC apprend des problemes non lineaires simples et plus difficiles. La partie Keras montre ensuite comment utiliser un framework deep learning pour passer a un dataset d'images standard comme MNIST.
