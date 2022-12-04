# Accessing the full policy space

## Introduction

Comme j'ai commencé à l'expliquer dans thirsty, et comme je vais continuer à l'expliquer dans un papier, avoir un choix d'action indépendant pour chaque robot, c'est juste pas possible. 

Plus précisément, le challenge, c'est de paramétriser une distribution sur beaucoup (beaucoup) d'actions différentes : Il nous faut un modèle génératif (pas pour faire du model-based, non, mais pour générer des actions). On a vu que les techniques récurrentes style pixel-CNN sont trop lentes. Du coup, on se tourne vers les méthodes de diffusion.

## Entraîner un modèle de diffusion

Mon test-case pour les modèles de diffusion est le suivant :

Imaginons qu'on a 25 robots disposés sur une grille de 5x5. La tâche pour ces robots, c'est que l'un d'entre eux prenne la parole. Le robot qui prend la parole doit être aléatoire uniforme mais il doit toujours y avoir exactement un et un seul robot qui prend la parole. 

Pour entraîner un modèle de diffusion à reproduire cette distribution, on commence par créer un dataset de cette distribution. Ensuite pour chaque exemple, on corrompt plus ou moins les exemples. On entraîne ensuite un réseau de neurone (l'acteur ou le modèle de diffusion) à prédire l'exemple qui a mené à la donnée corrompue.

Une fois le réseau entraîné, on peut l'utiliser pour générer un truc qui, avec un peu de chance, se trouve dans l'ensemble d'entraînement : on commence par une image totalement aléatoire. Et par étape, on prédit l'exemple qui a généré l'image, on bouge un peu dans la dirrection de cet exemple. Rinse and repeat.

Le problème que j'observe sur mon dataset tout simple, c'est que le modèle "dérive" hors de la distribution d'entraînement : pendant le sampling, le réseau de neurones fait des petites erreurs au début et tombe au fur et à mesure sur des distributions qu'il n'a pas ou peu observé pendant l'entraînement. Pour corriger ça, le truc que je veux tester, c'est de prendre une image générée à une étape intermédiaire du sampling et de calculer la "vraie" prediction qu'on devrait faire. Un petit problème, c'est que si on veut calculer ça pour n exemples et pour m images générées pendant le sampling, ça va prendre O(n*m) pour calculer les "vraies" prédictions target.

Bref, quand on a un petit dataset ça peut le faire, pk ça prend pas trop de temps... Mais si on a un dataset énorme, ça devient clairement prohibitif, surtout que la proba une fois qu'on commence à drifter est majoritairement due à deux ou trois exemples. Dans le cas où on a beaucoup de data, ça donne presque envie de construire un KD-tree pour récupérer la data la plus intéressante.

Dans notre cas où il y a quand même pas beaucoup de donnée, on peut faire la procédure entière :
- On sample avec le modèle
- On calcule pour chaque sample la vraie prédiction qu'on aurait du effectuer
- On update le réseau de neurone pour matcher la vraie prédiction.
