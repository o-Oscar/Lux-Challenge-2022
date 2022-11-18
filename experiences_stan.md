## Expériences

Après correction du code une réplication des résultats d'oscar a pu être obtenue. L'erreur venait d'une non correction d'actions pas au bon format dans une version ultérieur de luxai.

Par la suite tout a été fait en enlevant le pb du POWER. Le `survivor_move` dans ce cas fait bien bouger les robots jusqu'à la fin.

Les expériences suivantes ont eu pour but de faire "danser" les robots en leur donnant des objectifs de mouvement plus précis.

- `survivor_move_position` donne une reward de mouvement vers le centre quand un robot est excentré (sur l'axe gauche/droite)
- `survivor_move_time` donne une reward de mouvement vers la droite ou la gauche suivant le temps écoulé (toutes les 25 step cela change) => nécessité d'utiliser `complete` obs pour que les robots aient accès à l'information du temps

=> résultats : assez décevant. Les robots s'autodétruisent pour maximiser leur reward à court termes, et cela même avec une reward de mouvement 100 fois plus petite que la reward de mort (donc ce n'est vraiment PAS rentable de perdre un robot).
- Pour `position` au lieu d'aller vers le centre ils se contente de faire droite/gauche pour choper la reward une fois sur 2.
=> On converge cependant en se stabilisant à une reward "correcte".
- Pour `time`, ils vont vers la gauche comme le demande la première phase mais ne retourne pas ver la droite après, et bcp de robot meurt en étant "plaquer" contre le mur de gauche. J'ai essayé d'augmentre la taille des couches convolutionnelles (64 au lieu de 32), d'avoir une observation binaire sur le temps au lieu d'une sinusoide (basé sur mes cycles droite/gauche, pas les cycles jours/nuit).
=> Aucune convergence, ca marche pas du tout et on oscille principalement autour de la reward 0.


## SECONDES SALVE D'EXP

- Grâces aux nouveaux générateurs d'observation (notamment `position_time`), les réseaux apprennent à faire danser les robots en les faisant aller de gauche à droite sur des périodes de 25 step (un demi-cycles jours/nuit).


## TODO

Une réfacto du code est nécessaire pour les entraînements. Avoir 4 fichier à modifier (`init`, `agent`, `train` et `evaluate`) tout en gérant les bons import etc, c'est ULTRA fastidieux et permet facilement bcp d'erreur (par exemple quand j'ai dédupliquer `survivor`, j'ai pas titlé sur le coup que je devais changer les import partout pour choper le bon init). Un système de fichier de config clean pourraient vraiment aider, où tout l'agent est décider dedans, tout en ayant à avoir un seul train/evaluate pour tlm quitte à en avoir un relativement compliqué car devant gérer les cas de manières exhaustive