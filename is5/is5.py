import keras.optimizers
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.datasets import boston_housing
import os
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Данные
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
print(train_data.shape)
print(test_data.shape)
print(test_targets)

# Нормализация данных
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std


# Модель
def build_model():
    model = Sequential()
    model.add(Dense(64, activation='sigmoid', input_shape=(train_data.shape[1],)))
    # model.add(Dense(128, activation='sigmoid'))
    model.add(Dense(1))
    model.compile(optimizer=keras.optimizers.Adam(0.1), loss='mse', metrics=['mae'])
    return model


# # Без кросс-валидации
# model = build_model()
# H = model.fit(train_data, train_targets, epochs=50, batch_size=10, verbose=1, validation_data=(test_data, test_targets))
#
# loss = H.history['loss']
# val_loss = H.history['val_loss']
# mae = H.history['mae']
# val_mae = H.history['val_mae']
# epochs = range(1, len(loss) + 1)
#
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.savefig('is5/plot_without_cross_v.png')


# Проверка
k = 5
num_val_samples = len(train_data) // k
all_scores = []
for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate([
        train_data[:i * num_val_samples],
        train_data[(i + 1) * num_val_samples:]], axis=0)
    partial_train_targets = np.concatenate([
        train_targets[:i * num_val_samples],
        train_targets[(i + 1) * num_val_samples:]], axis=0)
    model = build_model()
    H = model.fit(partial_train_data, partial_train_targets, epochs=50, batch_size=10, verbose=1)
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=1)
    all_scores.append(val_mae)

    loss = H.history['loss']
    mae = H.history['mae']
    epochs = range(1, len(loss) + 1)

    plt.clf()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, mae, 'b', label='Mae')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('is5/plot_loss' + str(i) + '.png')
print(f'Среднее: {np.mean(all_scores)}')
print(f'Все значения: {all_scores}')
