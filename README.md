

# Lux AI

## Commandes utiles

Pour lancer un entraînement :
```
python -m bots.survivor.train -h
```

## Etapes cleef de developpement qui nous permettent de dire qu'on a fait des progrès

- Entraîner un bot à aller chercher du minerais sans mourir.
- Entraîner une IA à contrôller plusieurs bots pour produire 10 bots.
- Entraîner une IA à générer du lichen

# Idée générale

Entraîner dirrectement un gros agent à gagner une partie, c'est très long. Faisable, mais très long et très gourmand en ressources. En effet, dans la situation où on lance un agent de RL à optimiser le reward binaire gagner/perdre, le signal d'entraînement est TRES TRES sparse. Une game peut durer 1000 tours, il peut y avoir près d'une centaine d'unitées en jeu au cours de la partie et chaque unité à une bonne dizaine d'actions disponnible à tout instant. Tout ça correspond à un espace des trajectoires de taille à peu près (10^100)^1000 = 10^100.000 (à comparer à 10^80, le nombre d'atome dans l'univers). Savoir quelle action changer pour gagner un peu plus peut donc être très compliqué si on a aucune idée de la structure globale du jeu.

Structurons tout ça.

On a un jeu où le concept central est l'unité (grosse, petite ou usine dans une moindre mesure). Ce sont les unités qui peuvent agir sur le monde. On va supposer que si on a des stratégies efficaces pour faire faire des choses aux unités, alors on pourra composer ces stratégies pour avoir une chance de gagner des parties. Cette hypothèse est sans doute discutable, mais elle nous permet de commencer à travailler.

On va essayer de créer des réseaux de neurones qui paramétrisent des stratégies capables de "faire faire des trucs cool", ou skill aux différentes unités. La première chose, c'est de définir la liste de skill qui nous intéressent. En voici une première liste :

- Aller sur une case de ressources (glace ou minerai)
- Aller sur une usine et y déposer son chargement
- Faire des allers-retours pour alimenter une usine en 
- Ne pas écraser d'autres unités en se déplaçant
- Déblayer les débris sur un chemin ou une zone
- ...

On remarque que les unités doivent récupérer les skills dans un certain ordre : il faut d'abord savoir se déplacer pour ensuite savoir comment éviter les autres robots et comment déblayer une zone.

Pour entraîner un skill, on a bien envie d'utiliser du RL (c'est même le but du projet). Pour ça, la méthode la plus prometteuse est d'utiliser un algo on-policy style PPO. Au cours de cet entraînement, on génère tout plein de trajectoires qu'on va enregistrer. Ensuite, pour entraîner une nouvelle stratégie pour résoudre un nouveau skill on peut bootstrap en commençant par un coup de Q-learning sur les trajectoires qu'on a déjà avant de se lancer dans l'apprentissage d'un nouveau skill avec une méthode on-policy style PPO. 

L'intérêt d'une méthode comme ça, c'est qu'il n'est pas nécessaire de conserver toujours la même fonction de reward : on peut utiliser du Q-learning sur des trajectoires générées avec une stratégie quelconque et avec une fonction de reward quelconque. Il faudra mesurer à quel point ça nous permet d'accélérer l'apprentissage de nouveaux skills par rapport à un simple PPO sans pré-entraînement. 

Bref, pour résumer, on veut avoir des unités qui, quand on les envoit quelque part, font les bonnes actions. Après, il restera à décider où les envoyer. Mais comme Stan peut l'expliquer mieux que moi, si les actions de base sont bonnes, alors si on envoit les unités un peu partout, on aura déjà une stratégie qui se débrouille bien.

## Quoi implémenter pour commencer à s'amuser

Pour faire marcher du Q-learning, il va falloir sauvegarder toutes (ou une grande majorité) des parties jouées par nos agents. Il faut un moyen de savegarder proprement les parties pour pouvoir les revoir avec le viewer et pour pouvoir les charger à nouveau dans python pour les manipuler comme on veut.

Il va falloir se creuser la tête pour réussir à faire rentrer notre problème dans le framework classique du Q-learning et de PPO.

Il faut réfléchir deux secondes à l'organisation du code. On va utiliser des gros bouts de code à plein d'endroits différents (environements standards, fonctions de reward, découpage des observations vers les différents agents, ...). Il faut qu'on sache facilement où est chaque bout de code et il faut pas qu'on re-code 30 fois les mêmes choses.


## Problème de l'environnement

En général en RL, on a un environnement qu'on ne peut pas trop modifier. En particulier, ici, c'et un peu relou de faire des sénarios qui entraînent spécifiquement un seul type de comportement des unités. Du coup, si on cherche à entraîner une unité à combattre, on doit d'abord être capable de survivre suffisement longtemps et produire suffisement de ressources et suffisement longtemps pour pouvoir envoyer des unités tester des techniques de combat.

Du coup l'environnement d'entraînement sera assez compliqué : supposons qu'on cherche à entraîner le combat. Il faut alors un séquenceur qui commence par arriver au mid-game, et qui une fois au mid-game envoit quelques unités pour combattre. 

Conclusion, pour chaque skill qu'on cherche à apprendre, il faut à peu près le même environnement, mais un séquenceur différent, avec une fonction de reward différente. 

## Misc

Il y a un système de queue d'actions qu'on peut envoyer aux unités. Utiliser ce système permet d'utiliser moins d'énergie. 

La solution générale, c'est d'utiliser un RNN (LSTM, Transformer, ...) qui output exactement toutes les actions à envoyer (en vrai, c'est peut-être pas une mauvaise idée) et qui finit par output \<end> quand il a fini de choisir ses actions. 

Une solution intermédiaire peut exploiter la structure de l'environnement : On peut se dire qu'on utlise le système de queue que pour les déplacement. Du coup quand on choisit de se déplacer, on envoit toutes les actions pour se déplacer vers la destination choisie. Par contre, pour miner ou autre, on envoit les actions une par une.

Bref, c'est un peu compliqué à implémenter. Dans un premier temps, on va s'en passer.

# Approche "un robot un réseau"

## Représentation d'une stratégie avec un réseaux de neurones par petit robot

Si on pousse l'analogie entre les unités du jeu et des petits robots qui se balladent sur Mars, on a envie de donner à chaque unité le même réseau de neurone. Chaque unité reçoit des inputs centrés sur sa position (la topologie locale du terrain, sa distance aux différentes usines, son stock de ressources, ...) et utilise son réseau de neurone (dont les poids sont partagés) pour choisir ses actions. 

On verra ça peut-être dans un second temps mais on peut aussi ajouter un vecteur de pré-condition au réseau de neurones pour avoir une sorte d'input qui correspondrait à un "ordre", de manière à envoyer des ordres différents à chaque unité pour les synchroniser entre elles. Ces "ordres" pourraient être donnés par un gros réseau de neurone "chef" qui a comme input toute la carte et qui donne des informations ou des ordres à chaque robot. 

On remarque qu'un réseau de neurones qui output des Q-valeurs peut aussi output des probabilités en ajoutant un simple softmax. Du coup, ça nous permet d'utiliser le même réseau de neurones pour le Q-learning et pour l'apprentissage on-policy.

### Pour commencer

On va commencer avec un réseau de neurones qui output une distribution catégorique qui est un simple subset de toutes les actions disponnibles. On garde en tête qu'il faut mettre en place un système "d'action masking" qui empêche la sélection d'actions qui n'ont pas de sens (se déplacer à l'extérieur de la carte, transférer ou récupérer de l'énergie alors qu'il y a pas d'usine, transférer alors que le robot n'a rien sur lui...)

- [0] : Ne rien faire -> Ne pas envoyer d'action
- [1-4] : Bouger d'une case
- [5] : Transférer de l'ice à l'usine en dessous
- [6] : Transférer du minerais à l'usine en dessous
- [7-9] : Récupérer 50, 100 ou 150 énergie de l'usine
- [10] : Creuser

L'input du réseau de neurones va comprendre deux modalités : une grille et un vecteur. La grille contient les informations de terrain et le vecteur contient les informations du robot.

- La grille a 5 channels : une channel de glace, une channel de minerais, une channel de robots, une channel d'usines et un channel de out-of bounds.
- Le vecteur a 7 inputs : le delta de l'usine la plus proche, la quantité de glace, la quantité de minerais, la quantité d'énergie, l'indication de l'heure en sin/cos

## resultats

Avec une loss simple (un -1 si le robot meurt à ce tour), les robots apprennent bien à ne pas mourir.

Par contre il y a un soucis : Il y a au moins trois usine par team et 10 robot par usine, ça fait 60 robot en tout **dès le début de la partie**. Si chaque robot observe une zone de 11 par 11, soit 121 pixel par robot, ça veut dire qu'on doit processer $N_{obs} = 7260$ pixels à chaque tour. On peut comparer ça aux $N_{grid} = 48*48 = 2304$ pixels qu'on doit processer si on fait un gros réseau de neurone qui prend toute la map en entrée. Il faut aussi noter qu'au cours de la partie, on va probablement créer beaucoup plus de robots. Le déséquilibre entre ces deux $N$ va augmenter.

Bref, l'idée d'avoir un réseau de neurone par robot devient assez tendue dès qu'on commence à avoir beaucoup de robots (ce qui est le cas ici). On va donc changer de méthode de représentation des stratégies par réseau de neurones.

# Approche convolutionelle

## Pourquoi un réseau totalement convolutionnel

Comme on vient de le voir, c'est une mauvaise idée d'utiliser des réseaux de neurones complêtement séparés pour chaque robot. 

La technique qu'on cherche à trouver doit donc pouvoir utiliser de manière efficace la représentation sous forme d'image de la map, et ne doit pas nous laisser copier 100 fois les mêmes données pour calculer les actions des robots. 

Une approche qu'on peut donc défendre, qui concerve la possibilité de gérer un nombre variable de robots, et qui concerve aussi un degré de localité, c'est d'utiliser des réseaux de neurones totalemet convolutionnels : c'est comme si on appliquait un réseau de neurone pour chaque position de la map qui nous intéresse. Chaque robot se fait bien dicter ses actions par un réseau de neurone dont les poids sont partagés (les matrices de convolution) mais dont les observations sont locales. 

On peut voir trois soucis à cette approche. Le premier survient au début de l'aprentissage : quand tous les robots meurent et qu'il n'en reste que 4 ou 5 sur la carte, on calcule quand même les convolutions pour toute la carte, ce qui peut être vu comme une perte de temps. Cependant, ce problème disparaît dès que l'on a suffisemment de robots sur la carte (ce qui arrive assez vite, comme on l'a vu). 

Le deuxième soucis peut se trouver dans la représentation de l'état du jeu. On sait que chaque robot doit avoir accès à son état local (son cargo, sa quantité d'énergie, ...). Toutes ces informations doivent être mises dans des channels indépendantes de l'image de départ, ce qui créé une représentation très sparse de l'état du jeu. Encore une fois, ce problème disparaît quand la carte commence à être pleine de robots.

Enfin le dernier soucis est un soucis pratique d'implémentation : Quand chaque robot process se observations de manière indépendante, on peut envoyer à PPO des batchs avec des observations de timesteps différents. Mais si on doit utiliser un réseau de neurone sur toute la carte pour trouver les actions des robots pour un timestep, alors on est obligé d'envoyer toutes les infos d'un (ou plusieurs) timestep en même temps à PPO pour qu'il puisse travailler.

## Espace d'observations

L'espace d'observation va donc être une grosse image avec tout ce qu'il faut en channel pour contenir les mêmes informations que précédemment:

- une channel de glace
- une channel de minerais
- une channel de présence de robots
- une channel de présence d'usine
- une channel de la quantité de glace portée par le robot (0 sinon)
- une channel de la quantité de minerais portée par le robot (0 sinon)
- une channel de la quantité d'énergie portée par le robot (0 sinon)
- une channel de l'heure (sin)
- une channel de l'heure (cos)

## Implémentation et remarques

En lançant l'entrainement avec un réseau convolutionnel simple, on se rend vite compte qu'il y a un problème : Les agents ont des comportements très vite très déterministiques. Cette observation est un peu surprenante en sachant que le réseau de neurone a accès aux mêmes données que les réseaux de neurones de la précédente approche. Alors j'ai passé à peu près 3h à essayer de trouver des erreurs dans le code, mais rien n'y fait, on a toujours un comportement différent si on utilise un réseau de neurones à convolution simple qui prend en entrée toute la map.

Pour se rapprocher au plus de l'implémentation initiale (en gardant un ppo qui fonctionne avec une grille), j'ai re-structuré le réseau de neurones de manière à copier la structure du précédent réseau de neurones : Il y a un chemin du réseua de neurone qui utilise les observations "de carte" du robot et un chemin qui utilise les observations "d'état" du robot.

- Les observations de carte (les 4 premières channels) sont récupérées par un layer de convolution de taille 11 (même taille que la grille dans le ppo par agent).
- Les observations d'état (les 5 dernières channels) sont récupérées par un layer de convolution de taille 1. Il agit comme un fully connected sur les features locales.

Ensuite, on passe la concaténation des outputs de ces deux layers dans une série de layers de convolutions de taille 1. Avec cette structure de réseau de neurones, on retrouve les performances obtenues par la version précédente de ppo. Ci dessous, les courbes d'entraînement :

![alt text](rapports/imgs/capture.png)

Ce qui est intéressant de remarquer, c'est qu'avec un simple réseau de convolution, le réseau de neurone est parfaitement capable d'émuler le comportement du réseau de neurone double-chemin. Normalement, on s'attendrai à avoir de meilleurs performances avec un réseau de neurones plus puissant. Or c'est l'inverse qu'on observe ici : Avec un réseau de neurones qui prend plus d'informations en entrée, les performances sont moins bonnes. Et ce n'est pas un problème de données !! En effet, en entraînant plus longtemps, on génère plus de données (on joue plus de parties). Donc il ne peut pas y avoir de problème d'over-fitting. Cependant, on remarque que l'architecture simple (normalement plus puissante) n'arrive pas du tout à la même performance que le réseau double-chemins. Il y a donc un problème fondamental à PPO (et peut-être à toute une classe d'algos d'apprentissage par renforcement) qui l'empèche d'être efficace avec des réseaux de neurones trop gros. 

# TODO :

Créer un environnement pour commencer à entraîner des trucs

- [ ] Faire le ménage quand nécessaire
- [x] Retourner la carte d'observation (une par team).
- [x] Retourner la map de mask d'actions (une par team). 
- [x] Retourner la map de rewards (une par team). Les rewards sont distribués sur les cases des robots à l'observation de départ

Ecrire une fonction de reward pour différents skills :

- [x] Ne pas rester sur les cases de spawn, mais surtout ne pas se faire écraser.
- [ ] Ne pas écraser d'autres robots, mais surtout ne pas se faire écraser.
- [ ] Ne pas écraser les robots alliés, mais écraser les robots adverses !!
- [ ] Aller se mettre sur une case où il y a de l'eau
- [ ] Se charger d'eau à fond
- [ ] Recharger les usines en eau
- [ ] Aller se mettre sur une case où il y a du minerais
- [ ] Se charger de minerais à fond
- [ ] Recharger les usines en minerais


Le RL à proprement parler :

- [x] Recoder un algo pour entraîner des agents sur une grille
- [x] Entraîner des agents à ne pas se rentrer dedans.
- [ ] Entraîner des agents à se faire la guerre.
- [ ] Evaluer les gains par rapport à zéro entraînement.
- [ ] Faire un premier petit rapport
- [ ] Entraîner des agents à se mettre sur des cases d'eau
- [ ] Evaluer les gains par rapport à zéro entraînement
- [ ] Evaluer les gains entre : entraîner de zéro, utilier le modèle précédent, utiliser du Q-learning pour bootstrap l'entrainement avec la nouvelle fonction de reward.
- [ ] Faire un rapport. C'est déjà pas mal de boulot d'arriver ici. 
