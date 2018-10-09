import sys

import keras
import numpy as np
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential

if __name__ == '__main__':
    loud = False
    if len(sys.argv) > 1:
        loud = sys.argv[1].lower() == 'true'
    batch_size = 128
    epochs = 12

    x_train = np.load('MNIST/trainImages.npy').astype(np.float32) / 255.0
    y_train = np.load('MNIST/trainLabels.npy').astype(np.float32)
    x_test = np.load('MNIST/testImages.npy').astype(np.float32) / 255.0
    y_test = np.load('MNIST/testLabels.npy').astype(np.float32)
    keras_shape = (28, 28, 1)
    x_train = x_train.reshape((len(x_train),) + keras_shape)
    x_test = x_test.reshape((len(x_test),) + keras_shape)

    if loud:
        print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=keras_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
              validation_data=(x_test, y_test))
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print('Loss: {loss} \t Accuracy: {acc}'.format(loss=loss, acc=acc))
