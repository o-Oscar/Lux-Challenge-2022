

# Lux AI


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

## Représentation d'une stratégie par réseaux de neurones

Si on pousse l'analogie entre les unités du jeu et des petits robots qui se balladent sur Mars, on a envie de donner à chaque unité le même réseau de neurone. Chaque unité reçoit des inputs centrés sur sa position (la topologie locale du terrain, sa distance aux différentes usines, son stock de ressources, ...) et utilise son réseau de neurone (dont les poids sont partagés) pour choisir ses actions. 

On verra ça peut-être dans un second temps mais on peut aussi ajouter un vecteur de pré-condition aua réseau de neurones pour avoir une sorte d'input qui correspondrait à un "ordre", de manière à envoyer des ordres différents à chaque unité pour les synchroniser entre elles. Ces "ordres" pourraient être donnés par un gros réseau de neurone "chef" qui a comme input toute la carte et qui donne des informations ou des ordres à chaque robot. 

On remarque qu'un réseau de neurones qui output des Q-valeurs peut aussi output des probabilités en ajoutant un simple softmax. Du coup, ça nous permet d'utiliser le même réseau de neurones pour le Q-learning et pour l'apprentissage on-policy.

### Pour commencer

On va commencer avec un réseau de neurones qui output une distribution catégorique qui est un simple subset de toutes les actions disponnibles. On garde en tête qu'il faut mettre en place un système "d'action masking" qui empêche la sélection d'actions qui n'ont pas de sens (se déplacer à l'extérieur de la carte, transférer ou récupérer de l'énergie alors qu'il y a pas d'usine, transférer alors que le robot n'a rien sur lui...)

- [0] : Ne rien faire -> Ne pas envoyer d'action
- [1-4] : Bouger d'une case
- [5] : Transférer de l'ice à l'usine en dessous
- [6] : Transférer du minerais à l'usine en dessous
- [7-9] : Récupérer 50, 100 ou 150 énergie de l'usine
- [10] : Creuser

## Problème de l'environnement

En général en RL, on a un environnement qu'on ne peut pas trop modifier. En particulier, ici, c'et un peu relou de faire des sénarios qui entraînent spécifiquement un seul type de comportement des unités. Du coup, si on cherche à entraîner une unité à combattre, on doit d'abord être capable de survivre suffisement longtemps et produire suffisement de ressources et suffisement longtemps pour pouvoir envoyer des unités tester des techniques de combat.

Du coup l'environnement d'entraînement sera assez compliqué : supposons qu'on cherche à entraîner le combat. Il faut alors un séquenceur qui commence par arriver au mid-game, et qui une fois au mid-game envoit quelques unités pour combattre. 

Conclusion, pour chaque skill qu'on cherche à apprendre, il faut à peu près le même environnement, mais un séquenceur différent, avec une fonction de reward différente. 

## Misc

Il y a un système de queue d'actions qu'on peut envoyer aux unités. Utiliser ce système permet d'utiliser moins d'énergie. 

La solution générale, c'est d'utiliser un RNN (LSTM, Transformer, ...) qui output exactement toutes les actions à envoyer (en vrai, c'est peut-être pas une mauvaise idée) et qui finit par output \<end> quand il a fini de choisir ses actions. 

Une solution intermédiaire peut exploiter la structure de l'environnement : On peut se dire qu'on utlise le système de queue que pour les déplacement. Du coup quand on choisit de se déplacer, on envoit toutes les actions pour se déplacer vers la destination choisie. Par contre, pour miner ou autre, on envoit les actions une par une.

Bref, c'est un peu compliqué à implémenter. Dans un premier temps, on va s'en passer.





# TODO :

Créer un environnement pour commencer à entraîner des trucs

- [x] Faire un wrapper qui sauvegarde l'exécution de l'environnment pour pouvoir le visualiser et réutiliser les transitions si besoin.  
- [X] Implémenter un placement pas trop débile des usine histoire d'avoir une chance de survie
- [X] Les usines produisent toujours des petits robots dès que possible (dans un premier temps).
- [X] Effectuer les actions demandées par l'extérieur pour chacune des unitées. Les actions sont sous la forme d'un dictionnaire avec le nom de l'unité et son action. 
- [ ] Retourner le dictionnaire des observations des robots
- [ ] Retourner le dictionnaire des rewards. Chaque robot à un reward à soit qu'il essait d'optimiser.
- [ ] Retourner le dictionnaire des masks d'actions. 
- [ ] Implémenter un meilleur placement des usine pour augmenter les chances de survie


Ecrire une fonction de reward pour différents skills :

- [ ] Ne pas écraser d'autres robots, mais surtout ne pas se faire écraser.
- [ ] Aller se mettre sur une case où il y a de l'eau
- [ ] Se charger d'eau à fond
- [ ] Recharger les usines en eau
- [ ] Aller se mettre sur une case où il y a du minerais
- [ ] Se charger de minerais à fond
- [ ] Recharger les usines en minerais


Le RL à proprement parler :

- [ ] Trouver un algo pour entraîner un nombre variable d'agents
- [ ] Entraîner des agents à ne pas se rentrer dedans.
- [ ] Evaluer les gains par rapport à zéro entraînement.
- [ ] Faire un premier petit rapport
- [ ] Entraîner des agents à se mettre sur des cases d'eau
- [ ] Evaluer les gains par rapport à zéro entraînement
- [ ] Evaluer les gains entre : entraîner de zéro, utilier le modèle précédent, utiliser du Q-learning pour bootstrap l'entrainement avec la nouvelle fonction de reward.
- [ ] Faire un rapport. C'est déjà pas mal de boulot d'arriver ici. 
