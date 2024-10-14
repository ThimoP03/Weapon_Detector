import json  # Voor het laden en verwerken van JSON-bestanden, zoals annotaties
import logging  # Voor het loggen van informatie tijdens het proces
import os  # Voor bestands- en directorybeheer, zoals het controleren van bestanden en maken van mappen
import getpass  # Voor het ophalen van de gebruikersnaam van de huidige gebruiker, nuttig voor forensische logging
import platform  # Voor het ophalen van systeeminformatie, zoals het besturingssysteem, nuttig voor forensische logging
import time  # Voor tijdsgerelateerde functies zoals pauzes en tijdmetingen
from datetime import datetime  # Voor het werken met datums en tijden, vooral handig voor logging en tijdstempels
import numpy as np  # Voor numerieke berekeningen en het manipuleren van arrays en matrices
import cv2  # OpenCV-bibliotheek voor beeldverwerking, zoals het laden, bewerken en opslaan van afbeeldingen
from tqdm import tqdm  # Voor het tonen van voortgangsbalken tijdens het verwerken van grote hoeveelheden data
from utils import bereken_bestand_hash  # Een aangepaste functie om hashes van bestanden te berekenen voor forensische tracking

IMAGE_SIZE = (128, 128)

# Aangepaste logginginstellingen met gebruikers- en systeeminformatie
def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_bestandsnaam = os.path.join('__forensic_logs/ETL_PROCESS/ETL_Resize', f"log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
    logging.basicConfig(filename=log_bestandsnaam, level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')  # Correctie hier
    
    logging.info(f"ETL-proces gestart door gebruiker: {getpass.getuser()} op systeem: {platform.system()} {platform.release()}")

# Forensische logging van de hash van afbeeldingsbestanden vóór en na het resizen
def resize_image_and_annotations(image_path, annotation, new_size=IMAGE_SIZE):
    try:
        originele_hash = bereken_bestand_hash(image_path)
        logging.info(f"Originele afbeelding hash (SHA256): {originele_hash}")
        
        start_tijd = time.time()  # Voor prestatie-logging
        
        # Laad de afbeelding
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Fout: Kan afbeelding niet laden op {image_path}. Controleer het pad.")
        
        originele_hoogte, originele_breedte = image.shape[:2]

        # Resize de afbeelding
        resized_image = cv2.resize(image, new_size)

        # Bereken de schaalfactoren
        schaal_x = new_size[0] / originele_breedte
        schaal_y = new_size[1] / originele_hoogte

        # Pas segmentatiepunten aan
        segmentatie = annotation['segmentation'][0]  # Aangenomen dat er één polygoon per object is
        segmentatie_punten = np.array(segmentatie).reshape(-1, 2)
        resized_segmentatie = (segmentatie_punten * [schaal_x, schaal_y]).flatten().tolist()

        # Bereken nieuwe bounding box
        bbox = annotation['bbox']
        x, y, w, h = bbox
        resized_bbox = [x * schaal_x, y * schaal_y, w * schaal_x, h * schaal_y]

        eind_tijd = time.time()
        verwerkingstijd = eind_tijd - start_tijd
        logging.info(f"Afbeelding {image_path} succesvol resized in {verwerkingstijd:.4f} seconden.")

        resized_hash = bereken_bestand_hash(image_path)
        logging.info(f"Resized afbeelding hash (SHA256): {resized_hash}")
        
        return resized_image, resized_segmentatie, resized_bbox
    except Exception as e:
        logging.error(f"Er is een fout opgetreden bij het resizen van de afbeelding {image_path}: {e}")
        raise

# Aangepaste verwerkingsfunctie inclusief forensische logging
def process_json_annotations(annot_file, images_dir, output_dir, new_size=IMAGE_SIZE):
    try:
        start_tijd = time.time()
        logging.info(f"Start verwerking van annotaties voor resizen van afbeeldingen. Hash van het annotatiebestand: {bereken_bestand_hash(annot_file)}")

        # Lees de originele annotaties
        data = read_annots_json(annot_file)

        os.makedirs(output_dir, exist_ok=True)

        resultaten = []

        for image_info in tqdm(data['images'], desc="Afbeeldingen verwerken (Resizen)"):
            image_id = image_info['id']
            image_file = image_info['file_name']
            image_path = os.path.join(images_dir, image_file)

            # Zoek annotaties voor de huidige afbeelding
            annotaties = [ann for ann in data['annotations'] if ann['image_id'] == image_id]

            # Resize elke afbeelding en zijn annotaties
            resized_image, resized_segmentatie, resized_bbox = resize_image_and_annotations(
                image_path, annotaties[0], new_size
            )

            # Sla de resized afbeelding op
            resized_bestandsnaam = f"resized_{image_file}"
            resized_output_bestand = os.path.join(output_dir, resized_bestandsnaam)
            cv2.imwrite(resized_output_bestand, resized_image)
            logging.info(f"Resized afbeelding opgeslagen: {resized_output_bestand}. Hash van resized afbeelding: {bereken_bestand_hash(resized_output_bestand)}")

            # Opslaan van informatie over resized afbeeldingen
            for annotatie in annotaties:
                categorie_id = annotatie['category_id']
                oppervlakte = annotatie['area'] * (new_size[0] * new_size[1]) / (image_info['width'] * image_info['height'])  # Oppervlakte bijwerken
                resultaten.append({
                    "segmentation": resized_segmentatie,
                    "height": new_size[1],
                    "width": new_size[0],
                    "area": oppervlakte,
                    "bbox": resized_bbox,
                    "filename": resized_bestandsnaam,
                    "category_id": categorie_id
                })

        eind_tijd = time.time()
        verwerkingstijd = eind_tijd - start_tijd
        logging.info(f"Verwerking van annotaties voltooid in {verwerkingstijd:.4f} seconden.")
        
        return resultaten
    except Exception as e:
        logging.error(f"Er is een fout opgetreden tijdens de verwerking van annotaties: {e}")
        raise

# Functie om het JSON-annotatiebestand te lezen
def read_annots_json(jsonfile):
    try:
        with open(jsonfile, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logging.error(f"Er is een fout opgetreden bij het lezen van het JSON-bestand {jsonfile}: {e}")
        raise

# Functie om de aangepaste annotaties op te slaan
def save_updated_annotations(resultaten, output_annot, original_annot):
    try:
        # Lees de originele annotaties
        data = read_annots_json(original_annot)

        # Maak de directory aan als deze nog niet bestaat
        output_dir = os.path.dirname(output_annot)
        if not os.path.exists(output_dir) and output_dir != '':
            os.makedirs(output_dir)

        # Leeg de images en annotations lijst uit de originele data
        data["images"] = []
        data["annotations"] = []

        max_image_id = 0  # Begin met een nieuw ID voor afbeeldingen
        max_annotation_id = 0  # Begin met een nieuw ID voor annotaties

        for result in tqdm(resultaten, desc="Annotaties bijwerken (Resizen)"):
            segmentatie = result["segmentation"]
            hoogte = result["height"]
            breedte = result["width"]
            oppervlakte = result["area"]
            bbox = result["bbox"]
            bestandsnaam = result["filename"]
            categorie_id = result["category_id"]

            max_image_id += 1
            max_annotation_id += 1

            # Voeg de nieuwe afbeeldinginformatie toe
            nieuwe_afbeelding = {
                "id": max_image_id,
                "width": breedte,
                "height": hoogte,
                "file_name": bestandsnaam
            }
            data["images"].append(nieuwe_afbeelding)

            # Voeg de nieuwe annotatieinformatie toe
            nieuwe_annotatie = {
                "id": max_annotation_id,
                "iscrowd": 0,
                "image_id": max_image_id,
                "category_id": categorie_id,
                "segmentation": [segmentatie],
                "bbox": bbox,
                "area": oppervlakte
            }
            data['annotations'].append(nieuwe_annotatie)

        # Sla de bijgewerkte annotaties op in het uitvoerbestand
        with open(output_annot, 'w') as file:
            json.dump(data, file, indent=4)  # Met `indent=4` voor leesbaarheid

        logging.info(f"Bijgewerkte annotaties opgeslagen in {output_annot}")
    except Exception as e:
        logging.error(f"Er is een fout opgetreden bij het opslaan van de bijgewerkte annotaties: {e}")
        raise

# Run het script
if __name__ == '__main__':
    setup_logging("__forensic_logs/ETL_PROCESS/ETL_Resize")

    annot_file = '_ETL_DATA/Expanded_dataset/_coco_annots_train_expanded.json'
    images_dir = '_ETL_DATA/Expanded_dataset'
    output_dir = 'dataset/train/'
    output_annot = "dataset/train/annots/_train_annots.json"

    try:
        resultaten_resize = process_json_annotations(annot_file, images_dir, output_dir)

        save_updated_annotations(resultaten_resize, output_annot, annot_file)
    except Exception:
        print("Er is een fout opgetreden. Raadpleeg het logbestand voor meer informatie.")