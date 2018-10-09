import sys

import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential

SHAPE = (28, 28, 1)
LOUD = False
BATCH_SIZE = 128
EPOCHS = 12
pre_learn_weights = []
post_learn_weights = []
DATA_SET = 'MNIST'


def load_data():
    x_trn = np.load(DATA_SET + '/trainImages.npy').astype(np.float32) / 255.0
    y_trn = np.load(DATA_SET + '/trainLabels.npy').astype(np.float32)
    x_tst = np.load(DATA_SET + '/testImages.npy').astype(np.float32) / 255.0
    y_tst = np.load(DATA_SET + '/testLabels.npy').astype(np.float32)
    x_trn = x_trn.reshape((len(x_trn),) + SHAPE)
    x_tst = x_tst.reshape((len(x_tst),) + SHAPE)
    return x_trn, y_trn, x_tst, y_tst


def extract_weights():
    arr = np.array([])
    for layer in model.layers:
        for w in layer.get_weights():
            arr = np.append(arr, np.array(w).flatten())
    return arr


def mrs_labeled():
    pred = model.predict_classes(x_test)
    true_class = y_test.argmax(axis=1)
    incorrects = np.nonzero(pred != true_class)
    class_examples = [(incorrects[0][true_class[incorrects] == cls]) for cls in range(10)]
    for cls_ex in class_examples:
        if len(cls_ex):
            plt.imsave(DATA_SET + '_{}.png'.format(true_class[cls_ex[0]]),
                       x_test[cls_ex[0]].squeeze())


def plot():
    mn = min(np.min(pre_learn_weights), np.min(post_learn_weights))
    mx = max(np.max(pre_learn_weights), np.max(post_learn_weights))
    plt.hist(pre_learn_weights, label='Pre Training', range=(mn, mx), bins=1000, alpha=0.6)
    plt.hist(post_learn_weights, label='Post Training', range=(mn, mx), bins=1000, alpha=0.6)
    plt.legend()
    plt.savefig(DATA_SET + '_plt.png')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        LOUD = sys.argv[1].lower() == 'true'

    x_train, y_train, x_test, y_test = load_data()

    model = Sequential()
    model.add(Conv2D(20, kernel_size=(3, 3), activation='relu', input_shape=SHAPE))
    model.add(Conv2D(40, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(80, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    if LOUD:
        pre_learn_weights = extract_weights()

    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1,
              validation_data=(x_test, y_test))
    loss, acc = model.evaluate(x_test, y_test, verbose=0)

    print('Loss: {loss} \t Accuracy: {acc}'.format(loss=loss, acc=acc))
    if LOUD:
        mrs_labeled()
        post_learn_weights = extract_weights()
        plot()
