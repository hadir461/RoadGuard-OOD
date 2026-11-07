# RoadGuard-OOD
Road Damage Detection with YOLO11 and MLOps

Objectif : détecter les dégradations routières avec YOLO11n, étudier les données
ID/OOD et construire progressivement le suivi MLOps.

## Audit RDD2022

Depuis la racine du dépôt, avec Python 3.10 ou supérieur :

```powershell
python -m src.data.audit --input data/raw/RDD2022 --output reports/eda
python -m unittest discover -s tests -v
```

`--input` accepte un ou plusieurs dossiers. L'exploration est récursive ; les
chemins qui se recouvrent sont dédupliqués. Les extensions JPG/JPEG/PNG/XML sont
reconnues sans distinction de casse. India, Japan et Czech sont identifiés par
leurs dossiers ; les autres emplacements sont regroupés sous `unassigned`.
Les associations utilisent le même nom de base, sans distinction de casse,
dans un même périmètre (parent commun de `images/` et `annotations/`, normalement
`pays/train/`). Aucun appariement ne traverse les pays ou partitions existantes.
Un nom ambigu, par exemple deux extensions d'image pour un XML, est signalé.
Le champ XML `filename` est aussi comparé aux noms disponibles.

Le mode CLI par défaut, `--box-convention rdd2022`, accepte les bornes observées
dans les XML réels : `0 <= xmin < xmax <= width`, de même pour y. Par exemple,
`India_005123.xml` contient `(0, 373, 226, 720)` pour une image de 720 × 720,
et `Japan_000076.xml` contient `(0, 141, 47, 159)` pour 600 × 600. Les exemples
et compteurs de coordonnées sont conservés dans `coordinate_evidence` du JSON.
Les étendues géométriques sont `xmax-xmin` et `ymax-ymin` ; une étendue nulle
est signalée. Ce choix ne prouve pas l'inclusivité des pixels des annotations.
Le contrôle initial reste disponible avec `--box-convention voc` : origine 1,
bords inclus, étendues `xmax-xmin+1`. L'API Python `audit()` garde ce mode initial
et désactive le décodage par défaut pour préserver son comportement existant.

Chaque objet exige une boîte et des dimensions XML entières strictement positives.
Les instances D00/D10/D20/D40 incluent les boîtes invalides ; les statistiques de
boîtes les excluent. Les classes supplémentaires ne sont ni supprimées ni
remappées : elles figurent dans les anomalies et les fréquences de classes
inconnues. Les statistiques globales de boîtes incluent les classes inconnues ;
les statistiques par classe cible les séparent. Les XML sans objet sont comptés
à part et contribuent aux dimensions d'images si leur taille est valide.

Pillow, lorsqu'il est déjà installé, effectue `Image.open()` puis `load()` sur
chaque image, sans sauvegarde ni réparation. Les échecs et différences de taille
avec le XML sont signalés. Sinon, `image_integrity_checked` vaut `false` et le
statut indique explicitement l'absence de Pillow. `--skip-image-verification`
permet un audit des seules métadonnées. Aucun paquet n'est installé par le script.

Fichiers générés dans `--output` :

- `summary.json` : résumé global et par pays, anomalies détaillées et statistiques.
- `country_summary.csv` : fichiers, objets, associations et décodages par pays.
- `class_counts.csv` : instances et boîtes valides par pays et classe, plus classes inconnues.
- `dimension_statistics.csv` : effectif, minimum, maximum, moyenne et médiane des
  dimensions XML, dimensions/aires/ratios des boîtes et aires relatives ; aussi par classe cible.
- `anomalies.csv` : erreurs XML, dimensions, classes, boîtes, noms, doublons et décodage.
- `missing_associations.csv` : images sans XML et XML sans image, avec pays et partition.

Les CSV sont encodés en UTF-8 avec BOM pour Excel. Les statistiques de dimensions
d'images proviennent des XML, même lorsque le décodage est activé. Une statistique
sans observation a un effectif nul et des valeurs `null`, jamais des résultats inventés.

Les sources sont ouvertes en lecture seule. Le dossier de sortie doit être hors
des dossiers d'entrée. Chaque lancement actualise les six rapports dans ce dossier.
Un dossier vide produit un rapport à zéro avec
`dataset_found: false`. Les erreurs d'accès aux entrées donnent un code de sortie
1 ; les alertes de qualité XML sont consignées dans le rapport et n'empêchent pas
un audit terminé de retourner 0.

État actuel : audit des données RDD2022 extraites localement, sans modification
des données brutes. Aucun split, conversion YOLO, entraînement, résultat ID/OOD
ou service applicatif n'est encore produit. Czech reste un domaine d'évaluation
OOD et ne doit jamais servir à l'entraînement ou à la sélection du modèle.
Les 16 tests initiaux sont conservés ; les nouveaux tests utilisent uniquement
des XML et images synthétiques explicitement identifiés comme tels.
