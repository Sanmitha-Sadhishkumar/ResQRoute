import os
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, accuracy_score, log_loss, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# CNN Model Training
def cnn_train(model_filename, epochs):
    global cnn_model

    # Directory of the training dataset
    dataset_dir = '../datasets/adult_children_elderly/train'
    image_size = (640, 640)

    # Image Data Generator for loading and augmenting images
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.3)

    train_data_flow = datagen.flow_from_directory(
        dataset_dir,
        target_size=image_size,
        class_mode='categorical',
        batch_size=32,
        shuffle=True,
        subset='training'
    )

    validation_data_flow = datagen.flow_from_directory(
        dataset_dir,
        target_size=image_size,
        class_mode='categorical',
        batch_size=32,
        shuffle=False,
        subset='validation'
    )

    # Define the CNN model
    cnn_model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(640, 640, 3)),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax')  # Assuming 3 classes: Adult, Children, Elderly
    ])

    # Compile the model
    cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the CNN model
    history = cnn_model.fit(
        train_data_flow,
        validation_data=validation_data_flow,
        epochs=epochs,
        verbose=1
    )

    # Save the trained CNN model
    cnn_model.save(model_filename)
    print(f"Model saved to {model_filename}")
    plot_training_history(history)
    return f"Model saved to {model_filename}"

def plot_training_history(history):
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

from sklearn.metrics import roc_curve, auc
from itertools import cycle

def plot_roc_curves(y_true, y_pred_probs, n_classes):
    y_true_bin = tf.keras.utils.to_categorical(y_true, num_classes=n_classes)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.savefig('../static/cnn_roc.png')


def cnn_evaluate_on_test(model_filename):
    cnn_model = tf.keras.models.load_model(model_filename)

    test_dataset_dir = '../datasets/adult_children_elderly/test'
    image_size = (640, 640)

    test_datagen = ImageDataGenerator(rescale=1./255)

    test_data_flow = test_datagen.flow_from_directory(
        test_dataset_dir,
        target_size=image_size,
        class_mode='categorical',
        batch_size=32,
        shuffle=False
    )
    res=''
    test_steps = test_data_flow.samples // test_data_flow.batch_size
    results = cnn_model.evaluate(test_data_flow, steps=test_steps)
    print(f"Test Loss: {results[0]}")
    res+=f"Test Loss: {results[0]}\n"
    print(f"Test Accuracy: {results[1] * 100:.2f}%\n")
    res+=f"Test Accuracy: {results[1] * 100:.2f}%\n"

    y_pred_probs = cnn_model.predict(test_data_flow, steps=test_steps)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_data_flow.classes[:len(y_pred)]

    accuracy = accuracy_score(y_true, y_pred)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    logloss = log_loss(y_true, y_pred_probs)
    print(f"Log Loss: {logloss:.4f}")
    res+=f"Log Loss: {logloss:.4f}\n"

    class_labels = list(test_data_flow.class_indices.keys())
    report = classification_report(y_true, y_pred, target_names=class_labels)
    print("\nClassification Report:\n", report)
    res+=("\nClassification Report:\n"+ report)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plot_roc_curves(y_true, y_pred_probs, 3)
    plt.close()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig('../static/cnn_conf.png')
    return res

def preprocess_image(image_path, target_size=(640, 640)):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def predict_image_class_cnn(image_path, model_filename):
    cnn_model = tf.keras.models.load_model(model_filename)

    img_array = preprocess_image(image_path)

    prediction = cnn_model.predict(img_array)
    class_idx = np.argmax(prediction, axis=1)[0]

    class_labels = {0: 'Adult', 1: 'Children', 2: 'Elderly'}

    return class_labels[class_idx]
