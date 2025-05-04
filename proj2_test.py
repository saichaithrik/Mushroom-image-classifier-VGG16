import pandas as pd
import argparse
import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

IMG_SIZE = (224, 224)  # Match training image size

def load_model_weights(model_path):
    model = tf.keras.models.load_model(model_path)
    model.summary()
    return model

def get_images_labels(df, class_names, img_height=224, img_width=224):
    images = []
    labels = []

    for _, row in df.iterrows():
        img_path = row['image_path']
        label = row['label']

        # Load and preprocess image
        img = load_img(img_path, target_size=(img_height, img_width))
        img = img_to_array(img)
        img = img / 255.0  # Normalize to [0,1]
        images.append(img)
        labels.append(label)

    images = np.array(images)

    # Encode labels to integer indices
    label_encoder = LabelEncoder()
    label_encoder.fit(list(class_names))
    encoded_labels = label_encoder.transform(labels)
    categorical_labels = to_categorical(encoded_labels, num_classes=len(class_names))

    return images, categorical_labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transfer Learning Test Script")
    parser.add_argument('--model', type=str, default='mushroom_model_final.h5', help='Path to saved model file')
    parser.add_argument('--weights', type=str, default=None, help='Optional weights file')
    parser.add_argument('--test_csv', type=str, default='C:/Users/Sai Chaithrik/Desktop/Machine Learning project-2/sample_test_data/mushrooms_test.csv', help='Path to test CSV file')

    args = parser.parse_args()

    class_names = ['Agaricus', 'Amanita', 'Boletus', 'Cortinarius', 'Entoloma', 'Hygrocybe',
                   'Lactarius', 'Russula', 'Suillus']

    # Load CSV and prepend full image path if necessary
    test_df = pd.read_csv(args.test_csv)
    base_path = os.path.dirname(args.test_csv)
    test_df['image_path'] = test_df['image_path'].apply(lambda x: os.path.join(base_path, x))

    # Load model
    model = load_model_weights(args.model)

    # Optional: load weights separately
    if args.weights:
        model.load_weights(args.weights)

    # Get test data
    test_images, test_labels = get_images_labels(test_df, class_names)

    # Evaluate
    loss, acc = model.evaluate(test_images, test_labels, verbose=2)
    print('Test model accuracy: {:.2f}%'.format(acc * 100))

    # Predict and print for each image
    pred_probs = model.predict(test_images)
    pred_classes = np.argmax(pred_probs, axis=1)

    print("\nPer-image predictions:\n")
    for idx, pred in enumerate(pred_classes):
        image_file = os.path.basename(test_df.iloc[idx]['image_path'])
        true_label_index = np.argmax(test_labels[idx])
        true_class = class_names[true_label_index]
        predicted_class = class_names[pred]
        confidence = np.max(pred_probs[idx]) * 100

        print(f"[{idx+1}] Image: {image_file} | True: {true_class} | Predicted: {predicted_class} ({confidence:.2f}%)")
