# RoadGuard-OOD
Road Damage Detection with YOLO11 and MLOps

Objectif : détecter les dégradations routières avec YOLO11n, étudier les données
ID/OOD et construire progressivement le suivi MLOps.

## Inventaire minimal RDD2022

Depuis la racine du dépôt, avec Python 3.10 ou supérieur et sans dépendance externe :

```powershell
python -m src.data.audit --input data/raw --output reports/eda
python -m unittest discover -s tests -v
```

`--input` accepte un ou plusieurs dossiers. L'exploration est récursive, sans
structure imposée ; les fichiers de chemins qui se recouvrent sont comptés une
seule fois. `summary.json` contient les nombres d'images et de XML, les instances
D00/D10/D20/D40, les erreurs XML, les classes inconnues et les boîtes invalides.
Les extensions JPG/JPEG/PNG/XML sont reconnues sans distinction de casse.
Les images sont uniquement comptées par extension : leur intégrité et leurs
dimensions réelles ne sont pas vérifiées.

Convention des boîtes : coordonnées entières à origine 1 et bornes incluses,
`1 <= xmin <= xmax <= width` et `1 <= ymin <= ymax <= height`. Une boîte d'un pixel
est valide. Chaque objet doit avoir une boîte et le XML doit fournir une largeur
et une hauteur entières strictement positives. Les champs absents ou non entiers
sont signalés. Les instances de classes connues sont comptées même si leur boîte
est invalide ; les classes inconnues sont listées séparément. Les index d'objets
dans les alertes commencent à 1.

Les sources sont ouvertes en lecture seule. Le dossier de sortie doit être hors
des dossiers d'entrée. Chaque lancement actualise `summary.json` dans ce dossier.
Un dossier vide produit un rapport à zéro avec
`dataset_found: false`. Les erreurs d'accès aux entrées donnent un code de sortie
1 ; les alertes de qualité XML sont consignées dans le rapport et n'empêchent pas
un audit terminé de retourner 0.

État actuel : seul cet inventaire et ses tests synthétiques sont implémentés.
Aucun dataset RDD2022 n'est fourni ou téléchargé ; aucun entraînement, résultat
ID/OOD ou service applicatif n'est encore disponible. Les tests utilisent des XML
fabriqués et des fichiers images factices explicitement identifiés comme tels.
