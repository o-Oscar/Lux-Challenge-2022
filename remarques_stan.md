# Remarques Stan

Ce sont des remarques à chauds après avoir commencer à comprendre tout ce que tu as coder (tu codes plus vite que je lis xD)

## architectures fichiers :

Je trouve cela pas pratique/clair d'avoir à la fois du code dans `./bots/survivor` (notamment `env.py`) et dans `./utils/[action, obs, reward]`. De fait les fichiers `default` dedans sont spécifiques au bot survivor. Je propose donc de :

- soit garder `utils` (qui a de fait aussi un `env.py`) et chaque dossier [`action, obs, reward`] a un fichier pour chaque bot (par exemple `survivor.py`)
- soit on a un dossier pour chaque bot dans lequel on met 3 (4 ?) fichiers : [`action.py`, `obs.py`, `reward.py` (et ` env.py` ?) ], avec pourquoi pas un dossier `utils` mais qui sert de classe "mère" pour tous les bots.

## classes abstraites :

- Dans la continuité de ci-dessus : comme on va faire plein de tests, cela serait pas cool d'avoir à un endroit pour chaque type de fichier (`action`, `obs`, `reward`, etc) une classe "par défault" (avec pourquoi pas des fonction abstraites non implémentées), pour que ensuite chaque bot réimplémente ce qu'il veut.