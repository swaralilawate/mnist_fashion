import keras
import numpy as np
from keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

assert x_train.shape == (60000, 28, 28)
assert x_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)

x_test = x_test.astype("float32") / 255
x_test = np.expand_dims(x_test, -1)

model = keras.models.load_model("./trained_models/fashion_mnist_model.h5")
prediction = model.predict(x_test)
print("Number of images: ",len(prediction)) #same as number of images in x_test
print("Number of probabilities predicted: ", len(prediction[0])) #same as number of classes

for i in range(len(prediction)):
    print(f"Predicted label for {i}th image in x_test:",np.argmax(prediction[i]))
    print(f"Actual label for {i}th image in y_test:", y_test[i])

