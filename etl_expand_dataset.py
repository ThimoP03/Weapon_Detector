import json  # Voor het laden en verwerken van JSON-bestanden, zoals annotaties
import logging  # Voor het loggen van informatie tijdens het proces
import os  # Voor bestands- en directorybeheer, zoals het controleren en maken van mappen
import getpass  # Voor het ophalen van de huidige gebruikersnaam, nuttig voor forensische logging
import platform  # Voor het ophalen van systeeminformatie, zoals besturingssysteem, nuttig voor forensische logging
import time  # Voor tijdsgerelateerde functies, zoals pauzes en tijdmetingen
from tqdm import tqdm  # Voor het tonen van voortgangsbalken tijdens langdurige processen
from datetime import datetime  # Voor het werken met datums en tijdstempels, nuttig voor logging en forensische doeleinden
import random  # Voor willekeurige selectie en het werken met willekeurige waarden, handig voor bijvoorbeeld validatieprocessen
import numpy as np  # Voor numerieke berekeningen en het manipuleren van arrays en matrices
import cv2  # OpenCV-bibliotheek voor beeldverwerking, zoals het laden, bewerken en opslaan van afbeeldingen
from utils import bereken_bestand_hash  # Een aangepaste functie om hashes van bestanden te berekenen voor forensische tracking

# Aangepaste logginginstellingen met gebruikers- en systeeminformatie
def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_bestandsnaam = os.path.join('__forensic_logs/ETL_PROCESS/ETL_Expand', f"log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
    logging.basicConfig(filename=log_bestandsnaam, level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    logging.info(f"ETL-proces gestart door gebruiker: {getpass.getuser()} op systeem: {platform.system()} {platform.release()}")

# Forensische logging van de hash van afbeeldingsbestanden vóór en na het draaien
def rotate_image_and_annotations(image_path, annotation, image_width, image_height, angle):
    try:
        originele_hash = bereken_bestand_hash(image_path)
        logging.info(f"Originele afbeelding hash (SHA256): {originele_hash}")
        
        start_tijd = time.time()  # Voor prestatie-logging
        
        logging.info(f"Rotating image {image_path} by {angle} degrees")
        
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Error: Unable to load image at {image_path}. Please check the path.")
        
        # Extract bbox and segmentation from the annotation
        bbox = annotation['bbox']
        segmentation = annotation['segmentation'][0]  # Assuming one polygon per object

        # Reshape segmentation to be a list of (x, y) points
        segmentation_points = np.array(segmentation).reshape(-1, 2)

        # Get the center of the image for rotation
        center = (image_width / 2, image_height / 2)
        
        # Create rotation matrix for counter-clockwise rotation
        rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
        
        # Calculate new dimensions for the image to accommodate rotation
        abs_cos = abs(rotation_matrix[0, 0])
        abs_sin = abs(rotation_matrix[0, 1])
        new_width = int(image_height * abs_sin + image_width * abs_cos)
        new_height = int(image_height * abs_cos + image_width * abs_sin)

        # Adjust the rotation matrix to account for translation
        rotation_matrix[0, 2] += new_width / 2 - center[0]
        rotation_matrix[1, 2] += new_height / 2 - center[1]

        # Rotate the image
        image_rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))

        # Adjust segmentation points for the rotated image
        ones = np.ones(shape=(segmentation_points.shape[0], 1))
        points_ones = np.hstack((segmentation_points, ones))
        rotated_points = rotation_matrix.dot(points_ones.T).T

        # Convert rotated points back to a flat list
        rotated_segmentation = rotated_points.flatten().tolist()

        # Calculate new bounding box
        x, y, w, h = bbox
        bbox_points = np.array([[x, y], [x + w, y], [x, y + h], [x + w, y + h]])
        rotated_bbox_points = rotation_matrix.dot(np.hstack((bbox_points, np.ones((4, 1)))).T).T

        min_x = np.min(rotated_bbox_points[:, 0])
        min_y = np.min(rotated_bbox_points[:, 1])
        max_x = np.max(rotated_bbox_points[:, 0])
        max_y = np.max(rotated_bbox_points[:, 1])

        rotated_bbox = [min_x, min_y, max_x - min_x, max_y - min_y]
        new_area = rotated_bbox[2] * rotated_bbox[3]

        eind_tijd = time.time()
        verwerkingstijd = eind_tijd - start_tijd
        logging.info(f"Afbeelding {image_path} succesvol gedraaid in {verwerkingstijd:.4f} seconden.")

        rotated_hash = bereken_bestand_hash(image_path)
        logging.info(f"Rotated afbeelding hash (SHA256): {rotated_hash}")
        
        return image_rotated, rotated_segmentation, new_height, new_width, new_area, rotated_bbox
    except Exception as e:
        logging.error(f"An error occurred while rotating the image {image_path}: {e}")
        raise

# Functie om een willekeurige kleur te genereren in BGR-formaat
def generate_random_color():
    return [random.randint(0, 255) for _ in range(3)]  # Generate BGR color

# Functie om willekeurige kleuren toe te passen op de afbeelding
def apply_random_color(image, color):
    # Maak een gekleurd overlay
    colored_image = np.zeros_like(image)
    colored_image[:] = color
    # Mix de originele afbeelding met de gekleurde overlay
    return cv2.addWeighted(image, 0.5, colored_image, 0.5, 0)

# Functie om een enkele annotatie JSON te verwerken
def process_json_annotation_rotated(annot_file, images_dir, output_dir):
    try:
        logging.info("Starting annotation processing.")
        
        data = read_annots_json(annot_file)

        os.makedirs(output_dir, exist_ok=True)

        results = []

        for image_info in tqdm(data['images'], desc="Processing Images (Rotating)"):
            image_id = image_info['id']
            image_file = image_info['file_name']
            image_path = os.path.join(images_dir, image_file)
            image_width = image_info['width']
            image_height = image_info['height']

            annotations = [ann for ann in data['annotations'] if ann['image_id'] == image_id]

            for annotation in annotations:
                for i in range(10):  # Rotating by 30 degrees for 10 times
                    angle = (i + 1) * 30
                    rotated_image, rotated_segmentation, new_height, new_width, new_area, rotated_bbox = rotate_image_and_annotations(
                        image_path, annotation, image_width, image_height, angle
                    )

                    if rotated_image is not None:
                        # Save the original rotated image
                        filename = f"rotated_counter_{i + 1}x30_{image_file}"
                        output_file = os.path.join(output_dir, filename)
                        cv2.imwrite(output_file, rotated_image)
                        logging.info(f"Saved rotated image: {output_file}")

                        category_id = annotation['category_id']
                        results.append({
                            "rotated_segmentation": rotated_segmentation,
                            "new_height": new_height,
                            "new_width": new_width,
                            "new_area": new_area,
                            "rotated_bbox": rotated_bbox,
                            "filename": filename,
                            "category_id": category_id
                        })

        logging.info("Annotation processing completed.")
        
        return results
    except Exception as e:
        logging.error(f"An error occurred during annotation processing: {e}")
        raise

# Functie om JSON-annotaties te verwerken en willekeurige kleuren toe te passen
def process_json_annotation_colored(new_annot_file, new_images_dir, output_dir, num_colors=5):
    try:
        logging.info("Starting annotation processing for colored images.")
        
        data = read_annots_json(new_annot_file)

        os.makedirs(output_dir, exist_ok=True)

        results = []

        for image_info in tqdm(data['images'], desc="Processing Images (Coloring)"):
            image_id = image_info['id']
            image_file = image_info['file_name']
            image_path = os.path.join(new_images_dir, image_file)
            image_width = image_info['width']
            image_height = image_info['height']

            # Laad de afbeelding
            image = cv2.imread(image_path)
            if image is None:
                logging.warning(f"Unable to load image {image_path}. Skipping.")
                continue

            # Zoek annotaties voor de huidige afbeelding
            annotations = [ann for ann in data['annotations'] if ann['image_id'] == image_id]

            for j in range(num_colors):
                random_color = generate_random_color()
                colored_image = apply_random_color(image, random_color)

                # Sla de nieuwe gekleurde afbeelding op
                colored_filename = f"colored_{j + 1}_{image_file}"
                colored_output_file = os.path.join(output_dir, colored_filename)
                cv2.imwrite(colored_output_file, colored_image)
                logging.info(f"Saved colored image: {colored_output_file}")

                # Append colored image data to results for JSON
                for annotation in annotations:
                    segmentation = annotation['segmentation'][0]
                    area = annotation['area']
                    bbox = annotation['bbox']
                    category_id = annotation['category_id']

                    results.append({
                        "segmentation": segmentation,
                        "height": image_height,
                        "width": image_width,
                        "area": area,
                        "bbox": bbox,
                        "filename": colored_filename,
                        "category_id": category_id
                    })

        logging.info("Color processing completed.")
        
        # Return the results
        return results
    except Exception as e:
        logging.error(f"An error occurred during annotation processing: {e}")
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

# Functie om originele afbeeldingen te kopiëren naar een nieuwe directory
def copy_original_images(images_dir, new_dir):
    """
    Kopieert alle afbeeldingen van images_dir naar new_dir met behulp van alleen de os-bibliotheek,
    exclusief JSON-bestanden.
    
    Args:
        images_dir (str): Pad naar de directory met originele afbeeldingen.
        new_dir (str): Pad naar de nieuwe directory waar de afbeeldingen zullen worden gekopieerd.
    """
    try:
        # Maak de nieuwe directory aan als deze nog niet bestaat
        os.makedirs(new_dir, exist_ok=True)
        
        # Haal de lijst met alle bestanden in de originele afbeeldingsmap
        images = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]

        # Itereer over en kopieer elke afbeelding naar de nieuwe directory, exclusief JSON-bestanden
        for image in images:
            if image.endswith('.json'):
                continue  # Sla JSON-bestanden over
            
            src = os.path.join(images_dir, image)
            dst = os.path.join(new_dir, image)

            # Open het bronbestand en schrijf de inhoud naar het doelbestand
            with open(src, 'rb') as src_file:
                with open(dst, 'wb') as dst_file:
                    dst_file.write(src_file.read())
                    
            logging.info(f"Copied {image} to {new_dir}")

        logging.info(f"Alle afbeeldingen succesvol gekopieerd naar {new_dir}.")
    except Exception as e:
        logging.error(f"Error while copying images: {e}")
        print("Er is een fout opgetreden, raadpleeg het logbestand voor meer informatie.")

# Run het script
if __name__ == '__main__':
    setup_logging("__forensic_logs/ETL_PROCESS/ETL_Expand")
    
    annot_file = '_ETL_DATA\\Unprocessed_Images\\_coco_annots_train_unprocessed.json'
    images_dir = '_ETL_DATA\\Unprocessed_Images'
    output_dir = '_ETL_DATA\\Expanded_dataset'
    output_annot = "_ETL_DATA\\Expanded_dataset\\_coco_annots_train_expanded.json"

    try:
        logging.info("Starting rotation process.")
        
        results_rotate = process_json_annotation_rotated(annot_file, images_dir, output_dir)

        data = read_annots_json(annot_file)

        max_image_id = max(image['id'] for image in data['images'])

        for result in tqdm(results_rotate, desc="Updating Annotations (Resizing)"):
            new_segmentation = result["rotated_segmentation"]
            new_height = result["new_height"]
            new_width = result["new_width"]
            new_area = result["new_area"]
            rotated_bbox = result["rotated_bbox"]
            filename = result["filename"]
            category_id = result["category_id"]
            
            max_image_id += 1
            new_image = {
                "id": max_image_id,
                "width": new_width,
                "height": new_height,
                "file_name": filename
            }
            data["images"].append(new_image)
            
            new_annotation = {
                "id": max_image_id - 1,
                "iscrowd": 0,
                "image_id": max_image_id,
                "category_id": category_id,
                "segmentation": [new_segmentation],
                "bbox": rotated_bbox,
                "area": new_area
            }
            data['annotations'].append(new_annotation)

        with open(output_annot, 'w') as file:
            json.dump(data, file)
            
        logging.info(f"Rotation process completed. Annotations saved to {output_annot}")

        copy_original_images(images_dir, output_dir)
        
        logging.info("Starting color process.")
        
        results_color = process_json_annotation_colored(output_annot, output_dir, output_dir)

        data = read_annots_json(output_annot)

        max_image_id = max(image['id'] for image in data['images'])

        for result in tqdm(results_color, desc="Updating Annotations (Coloring)"):
            segmentation = result["segmentation"]
            height = result["height"]
            width = result["width"]
            area = result["area"]
            bbox = result["bbox"]
            filename = result["filename"]
            category_id = result["category_id"]
            
            max_image_id += 1
            new_image = {
                "id": max_image_id,
                "width": width,
                "height": height,
                "file_name": filename
            }
            data["images"].append(new_image)
            
            new_annotation = {
                "id": max_image_id - 1,
                "iscrowd": 0,
                "image_id": max_image_id,
                "category_id": category_id,
                "segmentation": [segmentation],
                "bbox": bbox,
                "area": area
            }
            data['annotations'].append(new_annotation)

        with open(output_annot, 'w') as file:
            json.dump(data, file)
        
        logging.info(f"Coloring process completed. Annotations saved to {output_annot}")
    except Exception:
        print("Er is een fout opgetreden, raadpleeg het logbestand voor meer informatie.")