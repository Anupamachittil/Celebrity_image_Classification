# Celebrity Image Classifier

This repository contains code for a Convolutional Neural Network (CNN) built with TensorFlow/Keras to classify images of celebrities. The model categorizes images into five classes representing different celebrities: Messi, Sharapova, Federer, Serena, and Kohli.

## Overview

The code performs the following steps:

1. **Data Preparation:**
   - Loads images of celebrities from respective folders.
   - Resizes images to a common size (128x128 pixels).
   - Stores image data and labels into lists (`data` and `label`).

2. **Data Preprocessing:**
   - Splits the dataset into training and testing sets (80-20 ratio).
   - Normalizes pixel values of the images.

3. **Model Architecture:**
   - Defines a CNN using Keras' Sequential model.
   - Utilizes convolutional layers, pooling layers, flattening, and fully connected layers.
   - Applies dropout regularization to prevent overfitting.

4. **Model Training:**
   - Compiles the model with Adam optimizer, sparse categorical crossentropy loss, and accuracy metrics.
   - Trains the model using training data for 25 epochs.

5. **Model Evaluation:**
   - Evaluates the trained model on the test data to calculate accuracy.
   - Generates a classification report showing precision, recall, and F1-score for each class.

6. **Prediction Functionality:**
   - Provides a function to predict the celebrity from a given image file using the trained model.

## Usage

### Prerequisites
- Python 3.x
- Required libraries: NumPy, Matplotlib, OpenCV, TensorFlow, PIL, scikit-learn

### Steps to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/celebrity-image-classifier.git
   cd celebrity-image-classifier

2. Install dependencies:
    ```bash
    pip install -r requirements.txt

3. Dataset Preparation:
    - Organize celebrity images into respective folders (lionel_messi, maria_sharapova, roger_federer, serena_williams, virat_kohli) within a directory named celebrities_data.

4. Run the code

## Model Accuracy
After training, the model achieved an accuracy of 82% on the test dataset.


