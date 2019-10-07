import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
from astropy.io.fits.tests import test_image
from tensorflow_core.examples.saved_model.integration_tests.mnist_util import INPUT_SHAPE

#print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels) = fashion_mnist.load_data()

class_names=['T-shirt/top','Trouser','pullover','Dress','Coat','Sandal','Shirt','Sneaker','bag','Ankle boot']

#print(train_images.shape)

#print(train_labels)

#print(test_images.shape, test_labels)

#first Img

def showImg(img):
    plt.figure()
    plt.imshow(train_images[img])
    plt.colorbar()
    plt.grid(False)
    plt.show()

#showImg(2000)

train_images = train_images/255.0

test_images = test_images /255.0

#Verifying the data
def showData():
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])
    plt.show()

#Model

model = keras.Sequential([
    keras.layers.Flatten(input_shape= (28,28)),
    keras.layers.Dense(128,activation = tf.nn.relu),
    keras.layers.Dense(10,activation = tf.nn.softmax)
])

#COmpile Model

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#trainning


model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images,test_labels)

#Test Accuracy: 0.81 < 0.91 de training model => Over-fitting
#print('Test accuracy: ', test_acc)

predictions = model.predict(test_images)

print(predictions[0])
print(np.array(predictions[0]))
print(test_labels[0])