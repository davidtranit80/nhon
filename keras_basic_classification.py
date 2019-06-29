# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from keras.models import model_from_json

# Helper libraries
import plotlib
import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt

# load data and divide dataset into training data and testing data
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# scale image data from (0,255) to (0,1)
train_images = train_images / 255.0
test_images = test_images / 255.0

# # build a sequence model with two hidden layers with 64 nodes and one ouput layer with 10 nodes
# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=(28, 28)),
#     keras.layers.Dense(64, activation=tf.nn.relu),
#     keras.layers.Dense(64, activation=tf.nn.relu),
#     keras.layers.Dense(10, activation=tf.nn.softmax)
# ])

# model.compile(optimizer='adam', 
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# model.fit(train_images, train_labels, epochs=10)

# # serialize model to JSON
# model_json = model.to_json()
# with open("basic_classification_model.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("basic_classification_weights.h5")
# print("Saved model_basic_classification to disk")

# # Predict test images
# test_loss, test_acc = model.evaluate(test_images, test_labels)

# print('Test accuracy:', test_acc)

# load json and create model
json_file = open('basic_classification_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("basic_classification_weights.h5")
print("Loaded model from disk")

predictions = loaded_model.predict(test_images)
# evaluate loaded model on test data
loaded_model.compile(optimizer='adam', 
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
test_loss1, test_acc1 = loaded_model.evaluate(test_images, test_labels)
print(f'Load model and Test Accuracy = {test_acc1}')
# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plotlib.plot_image(i, class_names, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plotlib.plot_value_array(i, predictions, test_labels)
plt.show()