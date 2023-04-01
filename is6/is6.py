import keras.optimizers
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.layers import Dense, Activation, Flatten
from keras.models import Sequential
import os
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
network_backup = "is6/network_backup"


# Получение данных
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Нормализация тестовых данных
train_images = train_images / 255.0
test_images = test_images / 255.0

# Нормализация меток
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


plt.figure(figsize=(10, 5))
for i in range(100):
    plt.subplot(10, 10, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(test_images[i], cmap=plt.cm.binary)
plt.show()


# plt.imshow(test_images[3], cmap=plt.cm.binary)
# plt.show()


if os.path.exists(network_backup):
    model_loaded = keras.models.load_model(network_backup)

    # Проверка своих данных
    # for i in range(10):
    #     img = cv2.cvtColor(cv2.imread(f'is6/custom_data_1/{i}.png'), cv2.COLOR_BGR2GRAY) / 255.0
    #     # plt.imshow(img, cmap=plt.cm.binary)
    #     # plt.show()
    #     x = np.expand_dims(img, axis=0)
    #     res = model_loaded.predict(x)
    #     print(f'correct = {i}, res = {np.argmax(res)}')
    i = 2
    img = cv2.cvtColor(cv2.imread(f'is6/custom_data_6/{i}.png'), cv2.COLOR_BGR2GRAY) / 255.0
    x = np.expand_dims(img, axis=0)
    res = model_loaded.predict(x)
    print(res)
    print(f'correct = {i}, res = {np.argmax(res)}')
else:
    # Архитектура сети
    model = Sequential()
    model.add(Flatten())
    # model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    # Подготовка к обучению
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Обучение
    # H = model.fit(train_images, train_labels, epochs=25, batch_size=100, validation_split=0.15)
    H = model.fit(train_images, train_labels, epochs=20, batch_size=100, validation_split=0.15)

    # Проверка
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('test_acc:', test_acc)

    # Получение ошибки и точности в процессе обучения
    loss = H.history['loss']
    val_loss = H.history['val_loss']
    acc = H.history['accuracy']
    val_acc = H.history['val_accuracy']
    epochs = range(1, len(loss) + 1)

    # Построение графика ошибки
    plt.clf()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('is6/plot_loss.png')

    # Построение графика точности
    plt.clf()
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('is6/plot_acc.png')

    # Проверка распознавания
    img = cv2.cvtColor(cv2.imread(f'is6/custom_data_1/3.png'), cv2.COLOR_BGR2GRAY) / 255.0
    x = np.expand_dims(img, axis=0)
    res = model.predict(x)
    print(res)
    print(f'correct = 3, res = {np.argmax(res)}')
    # model.save(network_backup)
