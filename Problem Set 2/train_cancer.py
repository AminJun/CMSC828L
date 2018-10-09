import sys

import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split

SHAPE = 9
LOUD = False
BATCH_SIZE = 1
EPOCHS = 100
pre_learn_weights = []
post_learn_weights = []
DATA_SET = 'Breast Cancer'


def load_data():
    x = np.genfromtxt(DATA_SET + '/breastCancerData.csv', delimiter=',')
    y = np.genfromtxt(DATA_SET + '/breastCancerLabels.csv', delimiter=',')
    return train_test_split(x, y, test_size=0.15)


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
            print('Miss labeled of class {} is {}'.format(cls_ex, x_test[cls_ex[0]].squeeze()))


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

    x_train, x_test, y_train, y_test = load_data()

    model = Sequential()
    model.add(Dense(units=16, activation='relu', input_dim=SHAPE))
    model.add(Dense(units=8, activation='relu'))
    model.add(Dense(units=6, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer=keras.optimizers.RMSprop(), loss=keras.losses.binary_crossentropy,
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
