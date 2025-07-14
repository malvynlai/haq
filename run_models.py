import argparse
import cv2
import numpy as np
import qai_hub as hub
from collections import Counter
from typing import List, Dict, Tuple


YOLO_MODEL_PATH = "yolov8_quantized.tflite" 
LLM_MODEL_PATH = "mistral_3b_quantized.tflite" 

TARGET_DEVICE = hub.Device("Apple M3 Air)")

COCO_CLASSES = {
}


def preprocess_image(image_path: str, input_size: Tuple[int, int] = (640, 640)) -> np.ndarray:
    """Loads and preprocesses an image for the YOLO model."""
    print(f"INFO: Preprocessing image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not find or open the image at: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize and pad
    h, w, _ = image_rgb.shape
    scale = min(input_size[0] / h, input_size[1] / w)
    resized_h, resized_w = int(h * scale), int(w * scale)
    resized_image = cv2.resize(image_rgb, (resized_w, resized_h))

    padded_image = np.full((input_size[0], input_size[1], 3), 128, dtype=np.uint8)
    padded_image[:resized_h, :resized_w] = resized_image

    # Normalize and expand dimensions for the model
    input_tensor = np.expand_dims(padded_image, axis=0).astype(np.float32)
    return input_tensor

def run_object_detection(model_path: str, device: hub.Device, input_tensor: np.ndarray) -> np.ndarray:
    """Runs the YOLO model using the Qualcomm AI Hub."""
    print(f"INFO: Submitting object detection job for model '{model_path}' on device '{device.name}'.")
    
    # The input to submit_inference_job is a dictionary.
    # The key ('image' in this case) must match the model's expected input name.
    inputs = dict(image=input_tensor)
    
    inference_job = hub.submit_inference_job(
        model=model_path,
        device=device,
        inputs=inputs
    )
    
    output_data = inference_job.download_output_data()
    
    # Output is a dictionary; we need to get the actual detection tensor from it.
    # The key ('output' here) depends on the model's output signature.
    detections = next(iter(output_data.values()))[0]
    return detections


def parse_yolo_output(detections: np.ndarray, confidence_threshold: float = 0.5) -> str:
    """Parses YOLO output to create a formatted list of food items."""
    print("INFO: Parsing object detection output.")
    
    detected_objects = []
    
    # Assuming output format is [x, y, w, h, confidence, class_id]
    for detection in detections:
        confidence = detection[4]
        if confidence >= confidence_threshold:
            class_id = int(detection[5])
            class_name = COCO_CLASSES.get(class_id, "unknown_object")
            detected_objects.append(class_name)
    
    if not detected_objects:
        return "No food items were detected in the fridge."
        
    item_counts = Counter(detected_objects)
    formatted_list = [f"{count} {item}(s)" for item, count in item_counts.items()]
    
    print(f"INFO: Detected items: {', '.join(formatted_list)}")
    return "I have the following ingredients: " + ", ".join(formatted_list) + "."


def run_recipe_generation(model_path: str, device: hub.Device, ingredients_prompt: str) -> str:
    """Uses an on-device LLM to generate a recipe."""
    print(f"INFO: Submitting recipe generation job for model '{model_path}'.")

    full_prompt = (
        f"{ingredients_prompt} "
        "Please provide a simple recipe I can make using these ingredients."
        "Assume I have basic pantry staples like oil, salt, and pepper."
    )
    
    inputs = dict(prompt=np.array([full_prompt]))
    
    inference_job = hub.submit_inference_job(
        model=model_path,
        device=device,
        inputs=inputs
    )
    
    output_data = inference_job.download_output_data()
    recipe = output_data['generated_text'][0] 
    return recipe


def main(image_path: str):
    """Main function to run the full fridge-to-recipe pipeline."""
    print("🚀 Starting Fridge-to-Recipe Pipeline 🚀")
    print("-" * 50)

    input_tensor = preprocess_image(image_path)
    detections = run_object_detection(YOLO_MODEL_PATH, TARGET_DEVICE, input_tensor)
    ingredients_prompt = parse_yolo_output(detections)
    
    if "No food items were detected" in ingredients_prompt:
        print("\n" + ingredients_prompt)
        return
        
    print("-" * 50)
    
    final_recipe = run_recipe_generation(LLM_MODEL_PATH, TARGET_DEVICE, ingredients_prompt)
    return final_recipe
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a recipe from an image of a fridge.")
    parser.add_argument(
        "image_path",
        type=str,
        help="The file path to the image of the fridge.",
    )
    args = parser.parse_args()
    
    main(args.image_path)

