import json  # Voor het laden en verwerken van JSON-annotatiebestanden
import logging  # Voor het loggen van informatie tijdens het proces
import os  # Voor bestands- en directorybeheer
import random  # Voor willekeurige selectie van afbeeldingen voor visualisatie
from datetime import datetime  # Voor het loggen van tijdstempels en forensische logging
import numpy as np  # Voor numerieke berekeningen en het manipuleren van arrays
import cv2  # OpenCV-bibliotheek voor het verwerken en bewerken van afbeeldingen
import matplotlib.pyplot as plt  # Voor het visualiseren van afbeeldingen en resultaten
from sklearn.model_selection import KFold  # Voor het uitvoeren van K-Fold Cross-Validation
import tensorflow as tf  # Voor het bouwen en trainen van diepe leermodellen
from tensorflow.keras import layers, models  # Voor het definiëren van lagen en het maken van Keras-modellen
from tensorflow.keras.callbacks import EarlyStopping  # Voor vroegtijdig stoppen van training om overfitting te voorkomen
from tqdm import tqdm  # Voor het weergeven van voortgangsbalken tijdens het verwerken van data
import getpass  # Voor het ophalen van de gebruikersnaam van de huidige gebruiker, nuttig voor forensische logging
import platform  # Voor het ophalen van systeeminformatie, zoals het besturingssysteem, nuttig voor forensische logging
from utils import bereken_bestand_hash  # Importeer je hash-functie voor forensische doeleinden

# Stel logging in (forensische logging)

def setup_logging(log_dir):
    # Zorg dat de directory voor logbestanden bestaat
    try:
        os.makedirs(log_dir, exist_ok=True)
        print(f"Log directory aangemaakt of al aanwezig: {log_dir}")
    except Exception as e:
        print(f"Fout bij het aanmaken van log directory: {e}")
    
    # Maak een logbestand aan met de huidige datum en tijd
    log_bestandsnaam = os.path.join(log_dir, f"log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
    
    try:
        with open(log_bestandsnaam, 'w') as f:
            f.write("")  # Check of bestand geopend kan worden
        print(f"Logbestand succesvol aangemaakt: {log_bestandsnaam}")
    except Exception as e:
        print(f"Fout bij het aanmaken van logbestand: {e}")
        return None

    # Stel een logger in
    logger = logging.getLogger("forensische_logger")
    logger.setLevel(logging.INFO)

    # Maak een handler aan voor het logbestand
    file_handler = logging.FileHandler(log_bestandsnaam, mode='a')
    
    # Stel een formatter in voor de logberichten
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Voeg de handler toe aan de logger
    logger.addHandler(file_handler)

    # Log informatie over de gebruiker en het systeem
    logger.info(f"Training gestart door gebruiker: {getpass.getuser()} op systeem: {platform.system()} {platform.release()}")
    
    return logger

logger = setup_logging("__forensic_logs/Training")  # Forensische logging setup

def flush_and_close_logger(logger):
    # Zorg ervoor dat alle loggegevens naar het bestand worden weggeschreven
    for handler in logger.handlers:
        handler.flush()
        handler.close()

# Controleren of TensorFlow gebruik maakt van GPU
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    logger.info(f"GPUs gedetecteerd: {len(physical_devices)}")
    for gpu in physical_devices:
        logger.info(f"Apparaat: {gpu}")
        details = tf.config.experimental.get_device_details(gpu)
        logger.info(f"GPU Details: {details}")
else:
    logger.warning("Geen GPU gedetecteerd. TensorFlow gebruikt CPU.")

# Functie voor het verwerken van annotaties en afbeeldingen
def process_annotations(annotations, img_dir):
    logger.info("Starten met het verwerken van annotaties...")
    images = []
    masks = []
    class_labels = []  # Opslag voor klasse labels
    image_names = []

    # Mapping van categorie ID naar namen (voor logger en tracking)
    category_names = {cat['id']: cat['name'] for cat in annotations['categories']}  
    logger.debug(f"Mapping van categorienamen: {category_names}")

    # Voortgangsbalk voor het verwerken van annotaties
    for img_info in tqdm(annotations['images'], desc="Verwerken van afbeeldingen en maskers"):
        img_path = os.path.join(img_dir, img_info['file_name'])
        img = cv2.imread(img_path)
        if img is None:
            logger.error(f"Afbeelding {img_info['file_name']} niet gevonden. Wordt overgeslagen.")
            continue

        # Haal hoogte en breedte van de afbeelding op
        img_height, img_width = img.shape[:2]
        anns = [ann for ann in annotations['annotations'] if ann['image_id'] == img_info['id']]

        # Maak een leeg masker voor alle categorieën
        mask = np.zeros((img_height, img_width), dtype=np.uint8)

        for ann in anns:
            segmentation = np.array(ann['segmentation'], dtype=np.float32).reshape(-1, 2)
            segmentation = segmentation.astype(np.int32)

            if segmentation.shape[0] == 0:
                logger.warning(f"Geen segmentatiepunten gevonden voor afbeelding {img_info['file_name']}, wordt overgeslagen.")
                continue

            try:
                # Vul het masker met objecten
                cv2.fillPoly(mask, [segmentation], 1)
            except Exception as e:
                logger.error(f"Fout tijdens fillPoly voor afbeelding {img_info['file_name']}: {e}")
                continue

            # Veronderstellen dat elke afbeelding één dominant object heeft met een klasse label
            class_labels.append(ann['category_id'] - 1)  # Verzamel klasse label (aangepast naar 0-index)

        images.append(img)
        masks.append(mask)
        image_names.append(img_info['file_name'])

    logger.info("Verwerken van annotaties voltooid.")
    return np.array(images), np.array(masks), np.array(class_labels), image_names

# Functie om willekeurige voorbeelden van afbeeldingen met maskers te visualiseren
def visualize_random_samples(img_dir, images, masks, image_names, num_samples=5):
    logger.info(f"Visualiseren van {num_samples} willekeurige voorbeelden.")
    indices = random.sample(range(len(images)), num_samples)

    for i in indices:
        img = images[i]
        mask = masks[i]
        file_name = image_names[i]

        # Constructeer het volledige pad van het oorspronkelijke bestand
        bestand_pad = f"{img_dir}/{file_name}"

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Bereken hash voor forensische doeleinden
        hash_value = bereken_bestand_hash(bestand_pad)

        # Maak een figuur met twee subplots (1 rij, 2 kolommen)
        plt.figure(figsize=(12, 6))
        
        # Stel een hoofdtitel in voor de gehele figuur
        plt.suptitle(f"Verificatie van dataset\nAfbeelding: {file_name}\nHash: {hash_value}", fontsize=16)

        # Oorspronkelijke afbeelding links (eerste subplot)
        plt.subplot(1, 2, 1)
        plt.imshow(img_rgb)
        plt.title("Oorspronkelijke afbeelding")
        plt.axis('off')

        # Masker rechts (tweede subplot)
        plt.subplot(1, 2, 2)
        plt.imshow(mask, cmap='gray')
        plt.title("Masker")
        plt.axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Layout aanpassen om de titel ruimte te geven
        plt.show()
        
    logger.info("Visualisatie voltooid.")

# Aangepast lichtgewicht model voor segmentatie (geen UNet)
def build_custom_segmentation_model(input_shape, num_classes=3):
    logger.info("Bouwen van aangepast segmentatiemodel gestart.")
    # Input laag
    inputs = layers.Input(shape=input_shape)

    # Gedeelde convolutionele lagen
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Tak voor segmentatiemasker
    mask_branch = layers.Conv2DTranspose(32, (3, 3), strides=2, padding='same', activation='relu')(x)
    mask_branch = layers.Conv2DTranspose(16, (3, 3), strides=2, padding='same', activation='relu')(mask_branch)
    mask_branch = layers.Conv2DTranspose(8, (3, 3), strides=2, padding='same', activation='relu')(mask_branch)
    mask_output = layers.Conv2D(1, (1, 1), activation='sigmoid', name='mask_output')(mask_branch)  # Sigmoid voor binair masker

    # Tak voor classificatie
    class_branch = layers.Flatten()(x)
    class_branch = layers.Dense(64, activation='relu')(class_branch)
    class_output = layers.Dense(num_classes, activation='softmax', name='class_output')(class_branch)  # Softmax voor klasse voorspelling

    # Maak het model met twee uitgangen: masker en klasse
    model = models.Model(inputs=inputs, outputs=[mask_output, class_output])
    logger.info("Model succesvol gebouwd.")
    return model

# Functie om trainingsgeschiedenis te plotten en op te slaan
def plot_and_save_history(history, fold):
    output_dir = 'Output_History'
    os.makedirs(output_dir, exist_ok=True)

    # Maak subplots voor verlies en nauwkeurigheid
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    # Plot training & validatie nauwkeurigheid
    axs[0].plot(history.history['class_output_accuracy'], label='Train Nauwkeurigheid')
    axs[0].plot(history.history['val_class_output_accuracy'], label='Val Nauwkeurigheid')
    axs[0].set_title(f'Fold {fold} - Model Nauwkeurigheid')
    axs[0].set_ylabel('Nauwkeurigheid')
    axs[0].set_xlabel('Epoch')
    axs[0].legend(loc='lower right')

    # Plot training & validatie verlies
    axs[1].plot(history.history['mask_output_loss'], label='Train Verlies (Mask)')
    axs[1].plot(history.history['val_mask_output_loss'], label='Val Verlies (Mask)')
    axs[1].plot(history.history['class_output_loss'], label='Train Verlies (Klasse)')
    axs[1].plot(history.history['val_class_output_loss'], label='Val Verlies (Klasse)')
    axs[1].set_title(f'Fold {fold} - Model Verlies')
    axs[1].set_ylabel('Verlies')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'history_fold_{fold}.png'))
    plt.close(fig)
    logger.info(f"Trainingsgeschiedenis voor Fold {fold} opgeslagen.")

# Functie om geschiedenis voor het finale model te plotten en op te slaan
def plot_and_save_final_model_history(history):
    output_dir = 'Output_History'
    os.makedirs(output_dir, exist_ok=True)

    # Maak subplots voor verlies en nauwkeurigheid
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    # Plot trainingsnauwkeurigheid
    axs[0].plot(history.history['class_output_accuracy'], label='Train Nauwkeurigheid')
    axs[0].set_title('Final Model - Train Nauwkeurigheid')
    axs[0].set_ylabel('Nauwkeurigheid')
    axs[0].set_xlabel('Epoch')
    axs[0].legend(loc='lower right')

    # Plot trainingsverlies
    axs[1].plot(history.history['mask_output_loss'], label='Train Verlies (Mask)')
    axs[1].plot(history.history['class_output_loss'], label='Train Verlies (Klasse)')
    axs[1].set_title('Final Model - Train Verlies')
    axs[1].set_ylabel('Verlies')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'final_model_history.png'))
    plt.close(fig)
    logger.info("Finale modeltrainingsgeschiedenis opgeslagen.")

# K-Fold Cross-Validation trainingsfunctie
def k_fold_cross_validation(images, masks, class_labels, k=5, epochs=10, batch_size=32, patience=2):
    """
    Voer K-Fold Cross-Validation uit om het beste model te selecteren op basis van validatiefout.
    Houdt forensische logger bij tijdens elke fold.
    """
    logger.info(f"Starten met K-Fold Cross-Validation met {k} folds.")
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    input_shape = (128, 128, 3)  # Verwacht invoerformaat van de afbeeldingen

    best_val_loss = float('inf')  # Houdt de laagste validatiefout bij om het beste model te selecteren
    best_model = None  # Houdt het beste model bij
    fold = 1

    # Itereer over elke fold (train op k-1 folds en valideer op de resterende fold)
    for train_index, val_index in tqdm(kf.split(images), total=k, desc="Cross-Validation Folds"):
        logger.info(f"Start Fold {fold}/{k}")
        
        # Train en validatiesets samenstellen
        train_images, val_images = images[train_index], images[val_index]
        train_masks, val_masks = masks[train_index], masks[val_index]
        train_labels, val_labels = class_labels[train_index], class_labels[val_index]

        # Normaliseer afbeeldingen naar [0, 1] en breid maskerdimensies uit
        train_images = train_images / 255.0
        val_images = val_images / 255.0
        train_masks = np.expand_dims(train_masks, axis=-1)
        val_masks = np.expand_dims(val_masks, axis=-1)

        # Bouw en compileer het segmentatiemodel
        model = build_custom_segmentation_model(input_shape, num_classes=len(np.unique(class_labels)))

        # Compileer het model
        model.compile(
            optimizer='adam',
            loss={'mask_output': 'binary_crossentropy', 'class_output': 'sparse_categorical_crossentropy'},
            metrics={'mask_output': 'accuracy', 'class_output': 'accuracy'}
        )

        # EarlyStopping callback instellen om overfitting te voorkomen
        early_stopping = EarlyStopping(
            monitor='val_loss',  # Valideer op basis van validatieverlies
            patience=patience,  # Stop na 'patience' aantal epochs zonder verbetering
            restore_best_weights=True  # Herstel de beste gewichten na vroegtijdig stoppen
        )

        # Train het model
        history = model.fit(
            train_images, 
            {'mask_output': train_masks, 'class_output': train_labels},
            steps_per_epoch=np.ceil(len(train_images) / batch_size).astype(int),
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(val_images, {'mask_output': val_masks, 'class_output': val_labels}),
            callbacks=[early_stopping]  # Voeg EarlyStopping toe aan het proces
        )

        # Controleer of Early Stopping is geactiveerd
        if early_stopping.stopped_epoch > 0:
            logger.info(f"Early Stopping geactiveerd in Fold {fold} na {early_stopping.stopped_epoch + 1} epochs.")
        else:
            logger.info(f"Early Stopping niet geactiveerd in Fold {fold}.")

        # Valideer en selecteer het beste model op basis van validatiefout
        val_loss = min(history.history['val_loss'])
        logger.info(f"Fold {fold} - Validatieverlies: {val_loss}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model  # Update het beste model op basis van validatieverlies
            logger.info(f"Nieuw beste model geselecteerd in Fold {fold} met validatieverlies {val_loss}")

        # Opslaan van modelgeschiedenis en model zelf
        plot_and_save_history(history, fold)
        model.save(f'models/model_fold_{fold}.h5')
        logger.info(f"Model voor Fold {fold} opgeslagen als 'model_fold_{fold}.h5'.")

        fold += 1
        tf.keras.backend.clear_session()  # Vrijgeven van geheugen na elke fold

    logger.info(f"Beste model geselecteerd met validatieverlies: {best_val_loss}")
    return best_model


# Hoofdfunctie om annotaties en afbeeldingen te laden en het model te trainen
if __name__ == '__main__':
    # Laad annotatie- en afbeeldingsgegevens
    annotations_file = 'dataset/train/annots/_train_annots.json'
    img_dir = 'dataset/train'

    logger.info("Laden van annotaties en afbeeldingen gestart.")
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)

    images, masks, class_labels, image_names = process_annotations(annotations, img_dir)
    
    # Hier worden 5 random images en hun maskers gevisualiseerd
    visualize_random_samples(img_dir, images, masks, image_names, num_samples=5)

    # K-Fold Cross-Validation uitvoeren en het beste model selecteren
    best_model = k_fold_cross_validation(images, masks, class_labels, k=5, epochs=5, batch_size=32)
    
    best_model.save('models/best_model.h5')  # Opslaan van het beste model

    logger.info("Training voltooid. Het beste model is geselecteerd en opgeslagen.")
    
    flush_and_close_logger(logger)