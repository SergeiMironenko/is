import keras.optimizers
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mclr
from keras import layers
from keras import models
import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
figpath = 'is2/'


def genData(size=500):
    size1 = size // 2
    size2 = size - size1
    t1 = np.random.rand(size1)
    x1 = np.asarray([i * math.cos(i * 2 * math.pi) + (np.random.rand(1) - 1) / 2 * i for i in t1])
    y1 = np.asarray([i * math.sin(i * 2 * math.pi) + (np.random.rand(1) - 1) / 2 * i for i in t1])
    data1 = np.hstack((x1, y1))
    label1 = np.zeros([size1, 1])
    div1 = round(size1 * 0.8)
    t2 = np.random.rand(size2)
    x2 = np.asarray([-i * math.cos(i * 2 * math.pi) + (np.random.rand(1) - 1) / 2 * i for i in t2])
    y2 = np.asarray([-i * math.sin(i * 2 * math.pi) + (np.random.rand(1) - 1) / 2 * i for i in t2])
    data2 = np.hstack((x2, y2))
    label2 = np.ones([size2, 1])
    div2 = round(size2 * 0.8)
    div = div1 + div2
    order = np.random.permutation(div)
    train_data = np.vstack((data1[:div1], data2[:div2]))
    test_data = np.vstack((data1[div1:], data2[div2:]))
    train_label = np.vstack((label1[:div1], label2[:div2]))
    test_label = np.vstack((label1[div1:], label2[div2:]))
    return (train_data[order, :], train_label[order, :]), (test_data, test_label)


def drawResults(data, label, prediction):
    plt.clf()
    p_label = np.array([round(x[0]) for x in prediction])
    plt.scatter(data[:, 0], data[:, 1], s=30, c=label[:, 0], cmap=mclr.ListedColormap(['red', 'blue']))
    plt.scatter(data[:, 0], data[:, 1], s=10, c=p_label, cmap=mclr.ListedColormap(['red', 'blue']))
    plt.grid()
    # plt.show()
    plt.savefig(figpath + 'plot_data.png')


# Получение начальных данных
(train_data, train_labels), (test_data, test_labels) = genData()


# ########################## #
# Создание и обучение модели #
# ########################## #


# Создание последовательной модели
model = models.Sequential()


# Добавление слоев
model.add(layers.Dense(units=128, activation='relu', input_shape=(2,)))
model.add(layers.Dense(units=16, activation='elu'))
model.add(layers.Dense(units=1, activation='sigmoid'))

# Настройки обучения модели
model.compile(optimizer=keras.optimizers.Adam(0.1), loss='mean_squared_error', metrics=['accuracy'])

# Обучение модели
H = model.fit(train_data, train_labels, epochs=20, verbose=1, validation_split=0.1)


# ################# #
# Вывод результатов #
# ################# #


# Получение ошибки и точности в процессе обучения
loss = H.history['loss']
val_loss = H.history['val_loss']
acc = H.history['accuracy']
val_acc = H.history['val_accuracy']
epochs = range(1, len(loss) + 1)

# Построение графика ошибки
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
# plt.show()
plt.savefig(figpath + 'plot_loss.png')

# Построение графика точности
plt.clf()
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
# plt.show()
plt.savefig(figpath + 'plot_acc.png')

# Получение и вывод результатов на тестовом наборе
results = model.evaluate(test_data, test_labels)
# print(results)

# Вывод результатов бинарной классификации
all_data = np.vstack((train_data, test_data))
all_label = np.vstack((train_labels, test_labels))
pred = model.predict(all_data)
drawResults(all_data, all_label, pred)
