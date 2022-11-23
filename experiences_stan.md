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


## TROISIEME SALVE D'EXP

- truc à retenir : faire gaffe à la période du sinus pour la reward (ne pas aller au delà)
- La danse est maintenant plus complexe ! On va en haut à gauche, puis à droite, puis en bas, puis à gauche
- Se préentrainer à ne pas mourir avant de danser permet de danser plus vite, mais ne permet pas d'avoir de meilleurs performances
- NB pour la reward : ce n'est pas vraiment un pb mais ca le sera peut-être plus tard : faut-il normaliser en fonction du nombre de factory au départ pour "lisser" la reward (si on a 2 fois plus de facto y a potentiellement 2 fois plus de destruction/de reward de mouvement)


## QUATRIEME SALVE D'EXP

Le but ici était d'avoir les robot collecter les resources. Le problème est la compétition entre eux, car ils ne savent pas à l'avance ce que leurs voisins vont faire. Une première solution est d'augmenter le `kernel_size` des dernières couches de l'agent, voir de dédoubler la dérnière couche (décision en 2 temps) Malheuresement cela ne suffit pas et ne permet que d'améliorer la vitesse d'apprentissage en début d'entrainement.
Consernant les résultat finaux : les robots se répartissent globalement sur les bonnes cases, mais "refusent" d'avoir des voisins, et se positionnent donc en diagonal les uns des autres, formant une disposition non optimale.