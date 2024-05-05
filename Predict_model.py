# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow import keras
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import warnings
warnings.filterwarnings('ignore')

print("Current working directory:", os.getcwd())

train_path = "train"
validation_path = "validation"
test_path = "test"

image_categories = os.listdir('C:/Users/User/Desktop/Project_322/train')


def plot_images(image_categories):
    # Create a figure
    plt.figure(figsize=(12, 12))
    for i, cat in enumerate(image_categories):
        # Load images for the ith category
        image_path = train_path + '/' + cat
        images_in_folder = os.listdir(image_path)
        first_image_of_folder = images_in_folder[0]
        first_image_path = image_path + '/' + first_image_of_folder
        img = image.load_img(first_image_path)
        img_arr = image.img_to_array(img) / 255.0

        # Create Subplot and plot the images
        plt.subplot(4, 5, i + 1)
        plt.imshow(img_arr)
        plt.title(cat)
        plt.axis('off')

    plt.show()


# Call the function
plot_images(image_categories)

# Creating Image Data Generator for train, validation, and test set

# 1. Train Set
train_gen = ImageDataGenerator(rescale=1.0/255.0)  # Normalize the data
train_image_generator = train_gen.flow_from_directory(
    train_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=True,)
# 2. Validation Set
val_gen = ImageDataGenerator(rescale=1.0/255.0)  # Normalize the data
val_image_generator = train_gen.flow_from_directory(
    validation_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False)

# 3. Test Set
test_gen = ImageDataGenerator(rescale=1.0/255.0)  # Normalize the data
test_image_generator = train_gen.flow_from_directory(
    test_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False)

# Print the class encodings done by the generators
class_map = dict([(v, k) for k, v in train_image_generator.class_indices.items()])
print(class_map)

# Build a custom sequential CNN model
model = Sequential()

# Add Layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Flatten the feature map
model.add(Flatten())

# Add the fully connected layers
model.add(Dense(32, activation='relu'))
model.add(Dense(5, activation='softmax'))    # Output_channel = 5

# Print the model summary
model.summary()

# Compile and fit the model
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)  # Set up callbacks
model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
hist = model.fit(train_image_generator,
                 epochs=100,
                 verbose=1,
                 validation_data=val_image_generator,
                 steps_per_epoch=3750//32,
                 validation_steps=750//32,
                 callbacks=early_stopping)

# Plot the error and accuracy
h = hist.history
plt.style.use('ggplot')
plt.figure(figsize=(10, 5))
plt.plot(h['loss'], c='red', label='Training Loss')
plt.plot(h['val_loss'], c='red', linestyle='--', label='Validation Loss')
plt.plot(h['accuracy'], c='blue', label='Training Accuracy')
plt.plot(h['val_accuracy'], c='blue', linestyle='--', label='Validation Accuracy')
plt.xlabel("Number of Epochs")
plt.legend(loc='best')
plt.show()

# Predict the accuracy for the test set
model.evaluate(test_image_generator)

print("Loss of the model is - ", model.evaluate(val_image_generator)[0])
print("Accuracy of the model is - ", model.evaluate(val_image_generator)[1]*100, "%")

from sklearn.metrics import confusion_matrix
import seaborn as sns

# Generate predictions for the validation data
y_pred = model.predict(val_image_generator)

# Convert predicted probabilities to class labels
y_pred_classes = np.argmax(y_pred, axis=1)

# Compute confusion matrix
conf_matrix = confusion_matrix(val_image_generator.classes[val_image_generator.index_array], y_pred_classes)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_map.values(), yticklabels=class_map.values())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Testing the Model
test_image_path = '3.jpg.jpg'
def generate_predictions(test_image_path, actual_label):
    # 1. Load and preprocess the image
    test_img = image.load_img(test_image_path, target_size=(224, 224))
    test_img_arr = image.img_to_array(test_img) / 255.0
    test_img_input = test_img_arr.reshape((1, test_img_arr.shape[0], test_img_arr.shape[1], test_img_arr.shape[2]))

    # 2. Make Predictions
    predicted_label = np.argmax(model.predict(test_img_input))
    predicted_vegetable = class_map[predicted_label]
    plt.figure(figsize=(4, 4))
    plt.imshow(test_img_arr)
    plt.title("Predicted Label: {}, Actual Label: {}".format(predicted_vegetable, actual_label))
    plt.grid()
    plt.axis('off')
    plt.show()


# call the function
generate_predictions(test_image_path, actual_label='Dollar')
