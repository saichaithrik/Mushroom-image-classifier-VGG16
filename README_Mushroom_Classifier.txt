
# 🍄 Mushroom Image Classifier using VGG16

This project uses **Transfer Learning** with the **VGG16** convolutional neural network to classify mushroom images into 9 distinct categories. The model is built using **TensorFlow** and **Keras**, and leverages data augmentation, early stopping, and fine-tuning techniques to achieve high classification accuracy.

---

## 📌 Project Overview

- **Goal**: Classify images of mushrooms into one of 9 species.
- **Approach**: Use pre-trained VGG16 model (from ImageNet) with a custom classification head.
- **Dataset**: Custom dataset organized into 9 folders (one per class), loaded using `ImageDataGenerator`.
- **Output**: Trained model saved in `.h5` format along with model weights.

---

## 🧠 Key Features

- ✅ Transfer Learning with VGG16 (frozen + fine-tuned layers)
- 🧪 Image Augmentation (rotation, zoom, flip, shift, shear)
- ⏹️ Early Stopping to prevent overfitting
- 🔁 ReduceLROnPlateau for dynamic learning rate adjustment
- 🧠 Classification over 9 mushroom categories
- 💾 Model and weights saved separately for reuse/deployment

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- VGG16 (from keras.applications)
- PIL (for image handling)
- NumPy

---

## 🚀 How to Run

1. **Clone the Repository**
   ```bash
   git clone https://github.com/saichaithrik/Mushroom-image-classifier-VGG16.git
   cd Mushroom-image-classifier-VGG16
   ```

2. **Set Up Virtual Environment (optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Place Your Dataset**
   - Dataset should be placed inside:
     ```
     /Mushrooms/Mushrooms/
     ├── Class1/
     ├── Class2/
     └── ...
     ```

5. **Run the Model**
   ```bash
   python mushroom_classifier.py
   ```

---

## 📊 Model Performance

- ✅ Training and validation split: 80-20
- 📈 Achieved high validation accuracy after fine-tuning
- 🧠 Final model saved as `mushroom_model_final.h5`
- 💾 Weights saved as `mushroom_weights_final.h5`

---

## 📁 Output Files

- `mushroom_model_final.h5`: Trained Keras model
- `mushroom_weights_final.h5`: Model weights
- `training_plot.png`: (Optional) Accuracy/Loss plots if generated

---

## 📎 LinkedIn Project Post

Check out my project highlight on LinkedIn:  
🔗 *Coming Soon*

---

## 📬 Contact

Feel free to reach out if you have questions or suggestions!

**Sai Chaithrik Dangeti**  
📧 saichaithrik@example.com  
🌐 [LinkedIn](https://linkedin.com/in/saichaithrik) | [GitHub](https://github.com/saichaithrik)

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).
