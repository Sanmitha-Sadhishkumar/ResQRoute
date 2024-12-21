import os
import numpy as np
import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, log_loss, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, cohen_kappa_score, confusion_matrix

def evaluate_random_forest_model(y_test, y_pred, y_prob):
    results = {}
    results['Accuracy'] = accuracy_score(y_test, y_pred)
    results['Precision'] = precision_score(y_test, y_pred, average='weighted')
    results['Recall'] = recall_score(y_test, y_pred, average='weighted')
    results['F1-Score'] = f1_score(y_test, y_pred, average='weighted')
    results['Confusion Matrix'] = confusion_matrix(y_test, y_pred)
    results['ROC-AUC'] = roc_auc_score(y_test, y_prob, multi_class='ovo', average='weighted')
    results['Cohen Kappa'] = cohen_kappa_score(y_test, y_pred)
    
    return results

X_train, X_test, y_train, y_test = [], [], [], []
features = []
labels = []
clf = ''
model_filename = '../models/random_forest_classifier.pkl'
result = ''

def rf_train(model_filename):
    global X_train, X_test, y_train, y_test, features, labels, clf, result

    dataset_dir = '../datasets/adult_children_elderly/train'
    image_size = (128, 128)

    datagen = ImageDataGenerator(rescale=1./255)
    data_flow = datagen.flow_from_directory(
        dataset_dir,
        target_size=image_size,
        class_mode='categorical',
        batch_size=32,
        shuffle=True
    )

    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)

    for batch in range(len(data_flow)):
        images, label_batch = next(data_flow)
        features_batch = model.predict(images)
        features.append(features_batch)
        labels.append(label_batch)

    features = np.vstack(features)
    features = features.reshape(features.shape[0], -1)
    labels = np.vstack(labels)

    y = np.argmax(labels, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.3, random_state=0)

    clf = RandomForestClassifier(n_estimators=125, criterion='gini', random_state=42)
    clf.fit(X_train, y_train)

    with open(model_filename, 'wb') as f:
        pickle.dump(clf, f)
    result = f"Model saved to {model_filename}\n"

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    result += f"Accuracy: {accuracy * 100:.2f}%\n"

    loss = log_loss(y_test, y_prob)
    result += f"Log Loss: {loss:.4f}\n"

    classification_rep = classification_report(y_test, y_pred)
    result += f"Classification Report:\n{classification_rep}\n"

    with open(model_filename, 'rb') as f:
        loaded_model = pickle.load(f)
    result += f"Model loaded from {model_filename}\n"

    y_loaded_pred = loaded_model.predict(X_test)
    loaded_accuracy = accuracy_score(y_test, y_loaded_pred)
    result += f"Loaded Model Accuracy: {loaded_accuracy * 100:.2f}%\n"

    result += print_model_params(clf)
    result += print_feature_importances(clf)
    confusion_matrix_file = plot_confusion_matrix(y_test, y_pred)
    result += f"Confusion Matrix saved to {confusion_matrix_file}\n"
    
    dict_result= evaluate_random_forest_model(y_test, y_pred, y_prob)
    for metric, value in dict_result.items():
        print(f"{metric}: {value}")
        result+=f"{metric}: {value}<br>"

    return result

def rf_test():
    global X_train, X_test, y_train, y_test, features, labels, clf, result
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    result = f"Accuracy: {accuracy * 100:.2f}%\n"

    loss = log_loss(y_test, y_prob)
    result += f"Log Loss: {loss:.4f}\n"

    classification_rep = classification_report(y_test, y_pred)
    result += f"Classification Report:\n{classification_rep}\n"

    with open(model_filename, 'rb') as f:
        loaded_model = pickle.load(f)
    result += f"Model loaded from {model_filename}\n"

    y_loaded_pred = loaded_model.predict(X_test)
    loaded_accuracy = accuracy_score(y_test, y_loaded_pred)
    result += f"Loaded Model Accuracy: {loaded_accuracy * 100:.2f}%\n"

    result += print_model_params(clf)
    result += print_feature_importances(clf)
    confusion_matrix_file = plot_confusion_matrix(y_test, y_pred)
    result += f"Confusion Matrix saved to {confusion_matrix_file}\n"

    return result

def det_rf(image_path):
    result=''
    model_filename = '../models/rf.pkl'
    with open(model_filename, 'rb') as f:
        clf = pickle.load(f)

    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)
    predicted_class = predict_image_class(image_path, model, clf)
    result = f"The predicted class for the image is: {predicted_class}\n"

    return result

def print_model_params(clf):
    output = "Random Forest Parameters:\n"
    output += f"Number of Estimators: {clf.n_estimators}\n"
    output += f"Criterion: {clf.criterion}\n"
    output += f"Max Depth: {clf.max_depth}\n"
    output += f"Min Samples Split: {clf.min_samples_split}\n"
    output += f"Min Samples Leaf: {clf.min_samples_leaf}\n"
    return output

def print_feature_importances(clf):
    output = "Feature Importances:\n"
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    for i in indices:
        output += f"Feature {i}: Importance {importances[i]}\n"
    return output

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    confusion_matrix_file = '../results/confusion_matrix.png'

    # Ensure the directory exists
    os.makedirs(os.path.dirname(confusion_matrix_file), exist_ok=True)
    
    plt.savefig(confusion_matrix_file)
    plt.close()
    return confusion_matrix_file

def predict_image_class(image_path):
    model_filename = '../models/rf.pkl'
    img = image.load_img(image_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)
    
    features = model.predict(img_array)
    features = features.reshape(features.shape[0], -1)

    with open(model_filename, 'rb') as f:
        clf = pickle.load(f)
    prediction = clf.predict(features)[0]
    class_labels = {0: 'Adult', 1: 'Children', 2: 'Elderly'}
    return class_labels[prediction]
