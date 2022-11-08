# Training agents to survive

## Introduction

Dans Lux AI, chaque joueur contrôle un tas de petits robots qui doivent aller chercher des ressources pour alimenter les centrales, ce qui permet in-fine de gagner la partie. A chaque tour, chaque joueur peut choisir une actions pour chacun de ses robots. Ces derniers peuvent se déplacer, creuser ou verser leur ressources dans l'usine la plus proche. En choisissant convenablement les actions, les joueurs parviennent à faire aller leurs robots sur des cases de ressource, à faire miner leurs robots et à rapporter les ressources aux usines. 

Nous essayons de contrôler nos robots avec des réseaux de neurones. Au début de l'entraînement, les réseaux de neurones proposent des actions aléatoire de manière totalement uniforme sur l'ensemble des actions disponnibles. Si on observe l'exécution d'une telle stratégie (la stratégie uniforme, ou non entraînée), on observe que la plupart des robots sont détruits au bout de quelques tours. Mais que s'est il passé !? 

Il se trouve que dans le jeu Lux AI, dans [les règles](https://www.lux-ai.org/specs-2022-beta) de résolution d'un tour, les robots peuvent se détruire les uns les autres en se déplaçant. Plus spécifiquement, si un robot se déplace sur la case d'un robot immobile, le robot immobile se fait détruire. Et si deux robots se déplacent le même tour sur la même case, les deux robots sont détruits.  Notons qu'il y a ce qu'on appelle dans le jeu vidéo le "friendly fire" : les règles de destruction entre robots s'appliquent que les robots fassent partie de la même équipe ou non. 

Ainsi, un premier skill à faire apprendre au réseau de neurone est d'éviter de faire collisionner les robots les uns avec les autres. Nous nous attardons donc dans ce mini-rapport sur notre méthode pour l'apprentissage de ce skill. 

## Représentation des stratégies par réseaux de neurones

Comme nous l'a appris l'expérience du communisme en URRS, la planification cntralisée de la production, c'est compliqué. 

Plus précisément, créer un réseau de neurones centrale qui coordone le mouvement de tous les robots, c'est plus compliqué que de donner à chaque robot un mini-réseau de neurones qui va se charger de sa survie individuelle. En effet, si chaque robot parvient à survivre, alors on aura réussi notre mission d'éviter les collisions entre robots. 

On donne donc à chaque robot un petit réseau de neurone. Les poids de ces réseaux de neurones sont partagés entre tous les robots, mais les observations sont unique pour chaque robot. Ainsi, chaque robot observe (entre autre) la carte et les autre robots autour de sa position. 

## Algorithme d'apprentissage

J'ai choisi d'utiliser l'algorithme PPO pour entraîner les poids du réseau de neurone des robots. Notons que l'implémentation classique de PPO suppose l'interraction entre un agent et un environnement. Des extensions ont été implémentées pour entraîner à la fois plusieurs agents en interraction avec un environnement. Cependant, je n'ai pas trouvé d'implémentation standard où le nombre d'agents en interraction avec l'environnement est variable (des robots sont créés et détruits au cours de l'exécution des parties). 

J'ai donc ré-implémenté PPO pour convenir à cette situation. 

Je pense que le choix spécifique de PPO est non critique, mais citons ici les deux raisons principales qui m'ont poussé à choisir PPO plutôt qu'un autre algorithme de RL. 

- Je connais bien PPO : J'ai réimplémenté PPO un grand nombre de fois pour savoir assez bien quels sont les détails d'implémentation et les problèmes de cet algorithme. Je remercie au passage Matteo Pirotta pour [le lien](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) vers un poste de blog qui compile les risques d'erreur d'implémentation de PPO. 
- PPO fait partie du State Of The Art (SOTA) au moment de l'écriture de ces lignes. C'est un algorithme model-free, on-policy, très pussant.

## Fonction de reward

Notons que le système de PPO multi-agent que j'ai implémenté permet de donner des rewards à chaque agent de manière indépendante : si un robot meurt c'est de sa faut à lui tout seul, pas à celui à l'autre bout de la map qui n'a rien fait de mal.

J'ai pris jusqu'ici un point de vue centré sur les robots individuels. La fonction de reward que j'ai implémenté suis ce même principe : chaque robot se voit attribuer une pénalité au moment de sa mort. 

## Entraînement

Chaque épisode correspond à l'exécution de la stratégie du réseau de neurone jusqu à ce que le jeu s'arrête. Dans notre cas, comme les usines ne reçoivent pas d'eau, la partie s'arrête après 100 tours. Il y a création de 10 robots par usine et entre 3 et 5 usines par joueur. Si tous les robots meurent après un tour (ce qui est possible), cela correspond au minimum à $2 \times 3 \times 10 = 60$ transitions utilisables par PPO pour cet épisode. A l'autre extrême, si tous les robots survivent jusqu'à la fin de la partie, on disposera au plus de $2 \times 5 \times 10 \times 100 = 10000$ transitions utilisables par PPO pour cet épisode.

A chaque episode, j'effectue quatre époques d'entraînement. 

A chaque époque, j'entraîne le réseau de neurone sur tout le dataset une fois en découpant le dataset en 4 mini-batch différents. 

J'entraîne le tout pour $\sim 500$ épisodes 

En 30 min on obtient une stratégie capable de préserver plus de 90% des robots à peu près tout le temps. 

## Découvertes intéressantes

### Sortie de l'usine

On observe que les robots ont deux dirrections privilégiées de sortie de l'usine. Ils ne se répartissent plus uniformément sur la carte qu'au bout d'une vingtaine de tours.

### Cycle jour-nuit

Il existe dans Lux AI un cycle circadien. De manière très intéressante, on remarque que les unités apprennent très vite à être moins active la nuit que le jour. Essay d'expliquer ce phénomène.

Pendant la journée les unités reçoivent une unité de puissance à chaque tour. La nuit ils n'en reçoivent pas. Se déplacer (avec l'implémentation actuelle des actions) coûte deux unités de puissance. Les unités commencent la partie avec cinquantes unités de puissance. 

Considérons le comportement d'une stratégie qui choisit ses action de manière aléatoire : Cette unité va perdre son énerge au fur et à mesure de la partie. Le jour elle pourra se déplacer car si elle ne bouge pas, ses batteries se remplissent toutes seules. La nuit cependant, avec des batteries vides, le robot ne pourra rien faire.

Ainsi, au début de l'entraînement, l'environnement est beaucoup moins dangereux la nuit que le jour car les robots bougent moins la nuit que le jour. PPO observe cette réalité statistique et décide alors de ne pas trop faire bouger les robots la nuit car (1) ils ne sont pas en danger et (2) cela leur permet d'économiser de la batterie. De cette manière, on obtient très rapidement un comportement où les robots bougent moins la nuit que le jour.