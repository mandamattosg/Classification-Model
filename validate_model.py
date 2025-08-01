import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configurations 
MODEL_PATH = "model/person_vs_horse_model.h5"  
VALIDATION_DIR = r"validation-horse-or-human"
OUTPUT_DIR = "validation_results" # New directory for results
IMG_SIZE = (224, 224)
CLASSES = {0: "horses", 1: "humans"} # Classes mapping

# Load the model
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    print(f"Error while loading the model: {e}")
    exit()


def preprocess_image(image_path):
    """Preprocess the image for prediction."""
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize(IMG_SIZE)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error while preprocessing {image_path}: {e}")
        return None

def run_validation():
    """
    Processes each image, makes predictions, and saves the results.
    It also calculates the accuracy of the model on the validation set.
    """
    correct_predictions = 0
    total_images = 0

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for class_name in os.listdir(VALIDATION_DIR):
        class_path = os.path.join(VALIDATION_DIR, class_name)
        if not os.path.isdir(class_path):
            continue

        true_label_id = list(CLASSES.keys())[list(CLASSES.values()).index(class_name)]
        
        image_files = [f for f in os.listdir(class_path) if f.endswith(('jpg', 'jpeg', 'png'))]

        for image_file in tqdm(image_files, desc=f"Processing images from '{class_name}'"):
            image_path = os.path.join(class_path, image_file)
            
            img_array = preprocess_image(image_path)
            if img_array is None:
                continue
            
            pred = model.predict(img_array, verbose=0)[0][0]
            
            predicted_label_id = 1 if pred > 0.5 else 0
            predicted_label = CLASSES[predicted_label_id]

            if predicted_label == "horses":
                confidence = 1 - pred  # Probability of being a horse
            else:
                confidence = pred # Probability of being a person

            
            is_correct = (predicted_label_id == true_label_id)
            if is_correct:
                correct_predictions += 1
            
            total_images += 1
            
            # Save the result as an image with prediction details
            img_display = Image.open(image_path).convert("RGB")
            
            plt.figure(figsize=(6, 6))
            plt.imshow(img_display)
            plt.axis("off")
            
            title_color = "green" if is_correct else "red"
            outcome_text = "Correct" if is_correct else "Incorrect"
            title_text = (
                f"{outcome_text}\n"
                f"True: {class_name}\n"
                f"Predicted: {predicted_label}\n"
                f"(Confidence: {confidence:.5f})"
            )
            plt.title(title_text, color=title_color)
            
            output_filename = f"{outcome_text.split()[0].lower()}_{image_file}"
            save_path = os.path.join(OUTPUT_DIR, output_filename)
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
            plt.close() 
           

    # Accuracy calculation    
    if total_images > 0:
        accuracy = (correct_predictions / total_images) * 100
        print(f"Images classified correctly: {correct_predictions}")
        print(f"General accuracy: {accuracy:.2f}%")
    else:
        print("No image found.")

if __name__ == "__main__":
    run_validation()