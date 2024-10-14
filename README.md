# Weapon Detector Thimo

Dit project is ontwikkeld door Thimo (s1142306) als onderdeel van een AI-model dat verschillende wapens detecteert en classificeert.

## Inhoud

- [Introductie](#introductie)
- [Installatie](#installatie)
- [Model Trainen](#model-trainen)
- [Model Gebruiken voor Voorspellingen](#model-gebruiken-voor-voorspellingen)

## Introductie

Dit project bevat een AI-model dat getraind is om verschillende wapens, zoals handvuurwapens en automatische wapens, te detecteren en te classificeren op basis van afbeeldingssegmentatie. Het model is getraind op een dataset met polygon-annotaties in COCO-formaat.

## Installatie

Volg de onderstaande stappen om het project lokaal in te stellen:

1. Clone de repository:
    ```bash
    git clone https://github.com/ThimoP03/Weapon_Detector_Thimo.git
    ```
2. Navigeer naar de projectmap:
    ```bash
    cd Weapon_Detector_Thimo
    ```
3. Installeer de vereiste Python-pakketten:
    ```bash
    pip install -r requirements.txt
    ```
4. Zorg ervoor dat je dataset zich in de map `_ETL_DATA/Unprocessed_Images` bevindt, met de polygon-annotaties in COCO-formaat.

## Model Trainen

Om het model opnieuw te trainen, volg deze stappen:

1. Run de datasetuitbreiding om de data voor te bereiden:
    ```bash
    python etl_expand_dataset.py
    ```
2. Run de script voor dataset-resizing:
    ```bash
    python etl_resize_dataset.py
    ```
3. Start het training script:
    ```bash
    python cnn_model_train.py
    ```

Het model wordt getraind op de beschikbare dataset en de resultaten worden opgeslagen in de bijbehorende outputmappen.

## Model Gebruiken voor Voorspellingen

Voor het maken van voorspellingen met het getrainde model, verwijs ik je door naar de Docker-gebaseerde oplossing in deze repo:

[Weapon Detector Docker](https://github.com/ThimoP03/Weapon_Detector_Docker)