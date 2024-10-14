# Weapon Detector Thimo
Dit model en project is gemaakt door Thimo (s1142306)

## Model opnieuw Trainen

Om het model opnieuw te trainen, zorg dat je deze git download. Vervolgens open de folder in een IDE naar keuze. Zorg dat de te gebruiken afbeeldingen in de folder _ETL_DATA/Unprocessed_Images komen te staan met de Polygon Annotaties (coco format). Vervolgens:

- Run etl_expand_dataset.py
- Run etl_resize_dataset.py
- Run cnn_model_train.py

Om predictions te kunnen doen, verwijs ik je door naar deze repo: https://github.com/ThimoP03/Weapon_Detector_Docker
