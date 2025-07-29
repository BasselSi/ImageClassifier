import argparse
import json
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from keras.utils import custom_object_scope
from PIL import Image

def load_model(model_path):
    # Specify the custom objects
    custom_objects = {'KerasLayer': hub.KerasLayer}
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    return model

def process_image(image_path):
    # Load and preprocess the image
    img = Image.open(image_path)
    img = img.resize((224, 224))  # Resize to match model's expected input
    img_array = np.array(img) / 255.0  # Normalize the image
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

def predict(image_path, model, top_k):
    # Process the image and make predictions
    processed_image = process_image(image_path)
    predictions = model.predict(processed_image)
    top_k_values = np.argsort(predictions[0])[-top_k:][::-1]
    return top_k_values, predictions[0][top_k_values]

def load_category_names(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description='Predict flower type from an image.')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('model_path', type=str, default='TensorFlow/intro-to-ml-tensorflow/lessons/intro-to-tensorflow/best_model.keras', help='Path to the saved model')

    parser.add_argument('--top_k', type=int, default=1, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, help='Path to a JSON file mapping labels to flower names')

    args = parser.parse_args()

    model = load_model(args.model_path)
    top_k_indices, top_k_probs = predict(args.image_path, model, args.top_k)

    if args.category_names:
        category_names = load_category_names(args.category_names)
        top_k_labels = [category_names[str(i)] for i in top_k_indices]
    else:
        top_k_labels = top_k_indices  # Use indices if no category names provided

    print("Top K Predictions:")
    for label, prob in zip(top_k_labels, top_k_probs):
        print(f"{label}: {prob:.4f}")

if __name__ == '__main__':
    main()
