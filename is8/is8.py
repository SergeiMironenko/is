from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.utils import np_utils
import keras
import numpy as np
import os
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

network_backup = "is8/network_backup"

if os.path.exists(network_backup):
    model_loaded = keras.models.load_model(network_backup)
    # Проверка своих данных
    for i in range(10):
        img = cv2.cvtColor(cv2.imread(f'is8/custom_data/{i}.png'), cv2.COLOR_BGR2GRAY) / 255.0
        # plt.imshow(img, cmap=plt.cm.binary)
        # plt.show()
        x = np.expand_dims(img, axis=0)
        res = model_loaded.predict(x)
        print(f'correct = {i}, res = {np.argmax(res)}')
else:
    (X_train, y_train), (X_test, y_test) = cifar10.load_data() # fetch CIFAR-10 data
    num_train, depth, height, width = X_train.shape # there are 50000 training examples in CIFAR-10
    num_test = X_test.shape[0] # there are 10000 test examples in CIFAR-10
    class_num = np.unique(y_train).shape[0] # there are 10 image classes
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= np.max(X_train) # Normalise data to [0, 1] range
    X_test /= np.max(X_train) # Normalise data to [0, 1] range

    y_train = np_utils.to_categorical(y_train, class_num) # One-hot encode the labels
    y_test = np_utils.to_categorical(y_test, class_num) # One-hot encode the labels

    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=X_train.shape[1:], padding='same'))
    model.add(Activation('relu'))

    model.add(Dropout(0.2))

    model.add(BatchNormalization())

    model.add(Conv2D(16, (3, 3), padding='same'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Conv2D(16, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Conv2D(16, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dropout(0.2))

    model.add(Dense(16, kernel_constraint=maxnorm(3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(16, kernel_constraint=maxnorm(3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Dense(class_num))
    model.add(Activation('softmax'))

    epochs = 5
    optimizer = 'adam'

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    print(model.summary())

    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=64)

    # Model evaluation
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    model.save(network_backup)