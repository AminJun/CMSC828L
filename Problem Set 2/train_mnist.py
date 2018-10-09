import sys

import keras
import numpy as np
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential

SHAPE = (28, 28, 1)
LOUD = False
BATCH_SIZE = 128
EPOCHS = 3


def load_data():
    x_trn = np.load('MNIST/trainImages.npy').astype(np.float32) / 255.0
    y_trn = np.load('MNIST/trainLabels.npy').astype(np.float32)
    x_tst = np.load('MNIST/testImages.npy').astype(np.float32) / 255.0
    y_tst = np.load('MNIST/testLabels.npy').astype(np.float32)
    x_trn = x_trn.reshape((len(x_trn),) + SHAPE)
    x_tst = x_tst.reshape((len(x_tst),) + SHAPE)
    return x_trn, y_trn, x_tst, y_tst


def mrs_labeled():
    pred = model.predict_classes(x_test)
    true_class = y_test.argmax(axis=1)
    incorrects = np.nonzero(pred != true_class)
    # examples = {}
    # for cls in range(0, 10):
    #     if len(incorrects[0][true_class[incorrects] == cls]):
    import pdb
    pdb.set_trace()


def plot():
    pass


if __name__ == '__main__':
    if len(sys.argv) > 1:
        LOUD = sys.argv[1].lower() == 'true'

    x_train, y_train, x_test, y_test = load_data()

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=SHAPE))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1,
              validation_data=(x_test, y_test))
    loss, acc = model.evaluate(x_test, y_test, verbose=0)

    print('Loss: {loss} \t Accuracy: {acc}'.format(loss=loss, acc=acc))
    if LOUD:
        mrs_labeled()
        plot()
