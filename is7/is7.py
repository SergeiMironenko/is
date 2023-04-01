import keras
import numpy as np
import math
from numpy import array
from keras.models import Model, Sequential
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
import csv
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Данные из варианта
nx, ne = 5, 3
x = np.linspace(start=0, stop=10, num=nx)
e = np.linspace(start=0, stop=0.3, num=ne)
data_in = []
data_out = []
for xi in x:
    for ei in e:
        data_in.append([math.cos(xi) + ei,
                  -xi + ei,
                  math.sqrt(math.fabs(xi)) + ei,
                  xi**2 + ei,
                  -math.fabs(xi) + 4,
                  xi - (xi**2) / 5 + ei])
        data_out.append(math.sin(xi) * xi + ei)

f = array(data_in)
fp = array(data_out)

n_in = len(f)
n_out = len(fp)

f = f.reshape((1, n_in, 6))
fp = fp.reshape((1, n_out, 1))
print(f)

# define encoder
visible = Input(shape=(n_in, 6))
encoder = LSTM(64, activation='relu')(visible)
encoderModel = keras.Model(visible, encoder)
encoderModel.compile(optimizer='adam', loss='mse')

# define reconstruct decoder
decoder1 = RepeatVector(n_in)(encoder)
decoder1 = LSTM(64, activation='relu', return_sequences=True)(decoder1)
decoder1 = TimeDistributed(Dense(6))(decoder1)
decoder1Model = keras.Model(Input(shape=(64, 1)), visible)
decoder1Model.compile(optimizer='adam', loss='mse')

# define predict decoder
decoder2 = RepeatVector(n_out)(encoder)
decoder2 = LSTM(64, activation='relu', return_sequences=True)(decoder2)
decoder2 = TimeDistributed(Dense(1))(decoder2)
decoder2Model = keras.Model(Input(shape=(64, 1)), visible)
decoder2Model.compile(optimizer='adam', loss='mse')

# tie it together
model = Model(inputs=visible, outputs=[decoder1, decoder2])
model.compile(optimizer='adam', loss='mse')

model.fit(f, [f, fp], epochs=300, verbose=1)
# demonstrate prediction
yhat = model.predict(f, verbose=0)
print(yhat[1])
print(fp)

# Сохранение данных
gen_data = open('is7/results/gen_data.csv', 'w+')
with gen_data:
    writer = csv.writer(gen_data, delimiter=' ', lineterminator='\n')
    writer.writerows([x, e])
    writer.writerows(data_in)
    writer.writerow(data_out)
gen_data.close()

np.asarray(encoderModel.predict(f)).tofile('is7/results/encode.csv', sep='\n')

np.asarray(yhat[0]).tofile('is7/results/decode.csv', sep='\n')
np.asarray(yhat[1]).tofile('is7/results/predict.csv', sep='\n')

encoderModel.save('is7/models/encoderModel.h5')
decoder1Model.save('is7/models/decoder1Model.h5')
decoder2Model.save('is7/models/decoder2Model.h5')
