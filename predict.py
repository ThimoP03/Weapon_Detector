import os  # Voor bestands- en mapbeheer
import numpy as np  # Voor numerieke operaties en arrays
import cv2  # OpenCV voor beeldverwerking
import logging  # Voor het loggen van activiteiten en fouten
from datetime import datetime  # Voor het werken met datums en tijden
import getpass  # Voor het verkrijgen van de huidige gebruikersnaam
import platform  # Voor het verkrijgen van informatie over het systeem
import tensorflow as tf  # TensorFlow voor machine learning modellen
from tensorflow.keras.models import load_model  # Voor het laden van een opgeslagen Keras-model
from utils import bereken_bestand_hash  # Functie voor het berekenen van een bestandshash voor forensische doeleinden


# Instellen van logging voor forensische doeleinden
def setup_logging(log_dir):
    # Zorg dat de directory voor logbestanden bestaat
    os.makedirs(log_dir, exist_ok=True)
    # Maak een logbestand aan met de huidige datum en tijd
    log_bestandsnaam = os.path.join('__forensic_logs/Predicting', f"log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
    logging.basicConfig(filename=log_bestandsnaam, level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Log informatie over de gebruiker en het systeem voor forensische doeleinden
    logging.info(f"Predictieproces gestart door gebruiker: {getpass.getuser()} op systeem: {platform.system()} {platform.release()}")

# Laad het getrainde model en log het proces
def load_trained_model(model_path):
    try:
        logging.info(f"Model wordt geladen van {model_path}")
        model = load_model(model_path)
        logging.info(f"Model {model_path} succesvol geladen.")
        return model
    except Exception as e:
        logging.error(f"Fout bij het laden van model {model_path}: {e}")
        raise

# Preprocess de afbeelding voor het doen van voorspellingen
def preprocess_image(image_path, target_size=(128, 128)):
    try:
        logging.info(f"Bezig met preprocessen van afbeelding: {image_path}")
        img = cv2.imread(image_path)
        if img is None:
            logging.error(f"Fout: Afbeelding {image_path} niet gevonden of kan niet worden geopend.")
            return None
        # Resize de afbeelding naar de vereiste inputgrootte (standaard 128x128)
        img = cv2.resize(img, target_size)
        img = img.astype(np.float32) / 255.0  # Normaliseer de afbeelding naar [0, 1]
        img = np.expand_dims(img, axis=0)  # Voeg een batchdimensie toe
        return img
    except Exception as e:
        logging.error(f"Fout tijdens preprocessen van afbeelding {image_path}: {e}")
        return None

# Postprocess de voorspelde maskers om bounding boxes te verkrijgen
def postprocess_mask(mask, threshold=0.5):
    try:
        logging.info(f"Bezig met postprocessen van masker met drempelwaarde {threshold}")
        # Binariseer het masker op basis van de drempelwaarde
        mask_bin = (mask > threshold).astype(np.uint8)
        # Zoek de contouren van de objecten in het masker
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Haal de bounding boxes van de gevonden contouren
        bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
        return bounding_boxes
    except Exception as e:
        logging.error(f"Fout tijdens het postprocessen van het masker: {e}")
        return []

# Teken de bounding boxes en mask op de afbeelding en sla deze op
def draw_predictions(image_path, mask, bounding_boxes, class_label, confidence_score, save_path, upscale_factor=4, font_scale=0.4, padding=10):
    try:
        logging.info(f"Voorspellingen worden getekend voor afbeelding: {image_path}")
        img = cv2.imread(image_path)

        if img is None:
            logging.error(f"Fout: Kan de afbeelding {image_path} niet openen of vinden.")
            return

        # Resize de originele afbeelding naar 128x128
        img_resized = cv2.resize(img, (128, 128))

        # Rescale het masker naar dezelfde grootte als de afbeelding
        mask_resized = cv2.resize(mask, (img_resized.shape[1], img_resized.shape[0]))
        mask_resized = (mask_resized * 255).astype(np.uint8)  # Zet het masker om naar schaal 0-255

        # Overlay het masker op de resized afbeelding
        img_with_mask = cv2.addWeighted(img_resized, 0.7, cv2.applyColorMap(mask_resized, cv2.COLORMAP_JET), 0.3, 0)

        # Verhoog de resolutie van de afbeelding voor betere leesbaarheid
        upscaled_image = cv2.resize(img_with_mask, (img_with_mask.shape[1] * upscale_factor, img_with_mask.shape[0] * upscale_factor), interpolation=cv2.INTER_CUBIC)

        # Voeg padding toe aan de upscaled afbeelding
        padded_image = cv2.copyMakeBorder(upscaled_image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        # Teken bounding boxes en plaats het classificatielabel en de score
        for bbox in bounding_boxes:
            x, y, w, h = bbox

            # Bereken de nieuwe coördinaten met padding
            x_padded = x * upscale_factor + padding
            y_padded = y * upscale_factor + padding

            cv2.rectangle(padded_image, (x_padded, y_padded), (x_padded + w * upscale_factor, y_padded + h * upscale_factor), (0, 255, 0), 2)

            # Voeg het label en de confidence score toe boven de bounding box
            label_text = f"{class_label}: {confidence_score:.2f}"
            cv2.putText(padded_image, label_text, (x_padded, y_padded - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)

        # Log informatie over de bounding boxes en de voorspelling
        logging.info(f"Voorspelde klasse: {class_label}, Confidence: {confidence_score:.2f}, Bounding Boxes: {bounding_boxes}")

        # Sla het resultaat op in het opgegeven pad
        cv2.imwrite(save_path, padded_image)
        logging.info(f"Afbeelding met voorspelling opgeslagen naar {save_path}")

    except Exception as e:
        logging.error(f"Fout tijdens het tekenen van voorspellingen voor afbeelding {image_path}: {e}")

# Definieer een mapping van klasse-indices naar klassennamen
CLASS_NAMES = {0: "Handgun", 1: "AutomaticWeapon", 2: "Knife"}  # Vervang met de daadwerkelijke klassennamen

# Voer voorspellingen uit voor alle afbeeldingen in een directory
def predict_on_directory(model, input_dir, output_dir, target_size=(128, 128)):
    logging.info(f"Start voorspellingen in directory: {input_dir}")
    # Zorg ervoor dat de output directory bestaat
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop door alle afbeeldingen in de input directory
    for image_name in os.listdir(input_dir):
        image_path = os.path.join(input_dir, image_name)

        # Log de hash van het bestand voor forensische tracking
        image_hash = bereken_bestand_hash(image_path)
        logging.info(f"Verwerken van afbeelding: {image_path}, Hash: {image_hash}")

        # Preprocess de afbeelding
        preprocessed_img = preprocess_image(image_path, target_size)
        if preprocessed_img is None:
            continue  # Sla over als de afbeelding niet gevonden of geopend kan worden

        # Doe de voorspelling (het model verwacht een batch)
        predictions = model.predict(preprocessed_img)
        
        # Haal de voorspellingen op: masker en klasse
        predicted_mask = predictions[0][0, :, :, 0]  # Eerste output is het masker
        class_predictions = predictions[1][0]  # Tweede output is de klasse-voorspelling
        
        # Haal de voorspelde klasse en bijbehorende confidence score
        predicted_class_index = np.argmax(class_predictions)
        predicted_class_label = CLASS_NAMES.get(predicted_class_index, "Onbekend")
        confidence_score = class_predictions[predicted_class_index]  # Confidence voor de voorspelde klasse

        # Postprocess het masker om bounding boxes te verkrijgen
        bounding_boxes = postprocess_mask(predicted_mask)

        # Log de uitkomst van de voorspelling
        logging.info(f"Voorspelde klasse voor {image_name}: {predicted_class_label}, Confidence: {confidence_score:.2f}")

        # Sla het voorspelde resultaat op in de output directory
        output_path = os.path.join(output_dir, image_name)
        draw_predictions(image_path, predicted_mask, bounding_boxes, predicted_class_label, confidence_score, output_path)

    logging.info(f"Voorspellingen opgeslagen naar {output_dir}")

if __name__ == "__main__":
    # Stel logging in
    setup_logging("__forensic_logs/Predicting")

    # Laad het getrainde model
    model_path = 'models/best_model.h5'
    model = load_trained_model(model_path)

    # Log de hash van het model voor forensische doeleinden
    model_hash = bereken_bestand_hash(model_path)
    logging.info(f"Model geladen met hash: {model_hash}")

    # Pad naar de directory met ongeziene afbeeldingen
    input_dir = 'Prediction_Images/todo'
    # Pad waar de voorspelde afbeeldingen met maskers en bounding boxes worden opgeslagen
    output_dir = 'Prediction_Images/done'

    # Voer voorspellingen uit voor alle afbeeldingen in de input directory en sla de resultaten op
    predict_on_directory(model, input_dir, output_dir)