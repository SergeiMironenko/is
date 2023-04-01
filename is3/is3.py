import pandas
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os
import keras.optimizers
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
figpath = 'is3/'

# Загрузка данных
dataframe = pandas.read_csv("is3/iris.csv", header=None)
dataset = dataframe.values
X = dataset[:, 0:4].astype(float)
Y = dataset[:, 4]

# Переход от текстовых меток к категориальному вектору
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
dummy_y = to_categorical(encoded_Y)

# Создание модели
model = Sequential()
# 1
model.add(Dense(64, activation='sigmoid'))
model.add(Dense(3, activation='softmax'))
# final
# model.add(Dense(64, activation='sigmoid', input_shape=(4,)))
# model.add(Dense(3, activation='sigmoid'))
# sigmoid, exponential

# Инициализация параметров обучения
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer=keras.optimizers.Adam(0.1), loss='categorical_crossentropy', metrics=['accuracy'])

# Обучение сети
H = model.fit(X, dummy_y, epochs=50, batch_size=10, validation_split=0.1)
# H = model.fit(X, dummy_y, epochs=100, batch_size=20, validation_split=0.1)


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
plt.savefig(figpath + 'plot_loss.png')

# Построение графика точности
plt.clf()
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(figpath + 'plot_acc.png')
