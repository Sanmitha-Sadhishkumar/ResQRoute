import os
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from sklearn.metrics import classification_report, accuracy_score, log_loss
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, cohen_kappa_score


def evaluate_model(y_test, y_pred, y_prob):
    results = {}
    results['Accuracy'] = accuracy_score(y_test, y_pred)
    results['Precision'] = precision_score(y_test, y_pred, average='weighted')
    results['Recall'] = recall_score(y_test, y_pred, average='weighted')
    results['F1-Score'] = f1_score(y_test, y_pred, average='weighted')
    results['Confusion Matrix'] = confusion_matrix(y_test, y_pred)
    results['ROC-AUC'] = roc_auc_score(y_test, y_prob, multi_class='ovo', average='weighted')
    results['Cohen Kappa'] = cohen_kappa_score(y_test, y_pred)
    return results

def dt_test():
  global X_train, X_test, y_train, y_test, features, labels, clf
  y_pred = clf.predict(X_test)
  y_prob = clf.predict_proba(X_test)
  result=''
  accuracy = accuracy_score(y_test, y_pred)
  result+=f"Accuracy: {accuracy * 100:.2f}%"
  print(f"Accuracy: {accuracy * 100:.2f}%")

  loss = log_loss(y_test, y_prob)
  result+=f"<br>Log Loss: {loss:.4f}"
  print(f"Log Loss: {loss:.4f}")
  result+=("<br>Classification Report:<br>", classification_report(y_test, y_pred))
  print("\nClassification Report:\n", classification_report(y_test, y_pred))

  tree_rules = export_text(clf)
  print("\nDecision Tree Rules:\n", tree_rules)
  return result

X_train, X_test, y_train, y_test = [], [], [], []
features = []
labels = []
clf=''
def dt_train(model_filename):
  global X_train, X_test, y_train, y_test, features, labels, clf
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

  clf = DecisionTreeClassifier(criterion='gini')
  clf.fit(X_train, y_train)

  tree_rules = export_text(clf)
  print("\nDecision Tree Rules:\n", tree_rules)
  result=("<br/>Decision Tree Rules:<br/>"+ str(tree_rules))
  with open(model_filename, 'wb') as f:
    pickle.dump(clf, f)
  result+=f"Model saved to {model_filename}"
  y_pred = clf.predict(X_test)
  y_prob = clf.predict_proba(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  result+=f"Accuracy: {accuracy * 100:.2f}%"
  print(f"Accuracy: {accuracy * 100:.2f}%")

  loss = log_loss(y_test, y_prob)
  result+=f"<br>Log Loss: {loss:.4f}"
  print(f"Log Loss: {loss:.4f}")
  result+=("<br>Classification Report:<br>"+ classification_report(y_test, y_pred)+'<br>')
  print("\nClassification Report:\n", classification_report(y_test, y_pred))
  evaluation_results = evaluate_model(y_test, y_pred, y_prob)
  for metric, value in evaluation_results.items():
    print(f"{metric}: {value}")
    result+=f"{metric}: {value}<br>"

  tree_rules = export_text(clf)
  print("\nDecision Tree Rules:\n", tree_rules)
  return result

def preprocess_image(image_path, target_size=(128, 128)):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def extract_features(img_array, model):
    features = model.predict(img_array)
    features = features.reshape(1, -1)
    return features

def predict_image_class(image_path, clf, model):
    img_array = preprocess_image(image_path)
    features = extract_features(img_array, model)
    class_idx = clf.predict(features)[0]
    class_labels = {0: 'Adult', 1: 'Children', 2: 'Elderly'}
    return class_labels[class_idx]

def dec_dt(image_path):
    model_filename = '../models/dt_gini.pkl'
    with open(model_filename, 'rb') as f:
        clf = pickle.load(f)

    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)
    predicted_class = predict_image_class(image_path, clf, model)
    print(f"The predicted class for the image is: {predicted_class}")
    return predicted_class