import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
# from keras import layers
# from tensorflow import keras
# from tensorflow.keras import layers
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Activation, Lambda, Add, Concatenate, Dropout

from keras.models import Model, load_model
from keras.preprocessing import image
from keras.initializers import glorot_uniform
from keras.utils import to_categorical
from keras import backend as K
from keras.callbacks import TensorBoard

from time import time
import os

sess = tf.Session()
K.set_session(sess)

DATA = './Data/'
MODEL = './Model'
path = os.path.join(MODEL, 'weights.{epoch:02d}-{loss:.4f}-{acc:.4f}.hdf5')

def load_data():
	data_train = pd.read_csv(os.path.join(DATA, 'train.csv'))
	data_test = pd.read_csv(os.path.join(DATA, 'test.csv'))

	img_rows, img_cols = 28, 28
	input_shape = (img_rows, img_cols, 1)

	X = np.array(data_train.iloc[:, 1:])
	y = to_categorical(np.array(data_train.iloc[:, 0]))

	#Here we split validation data to optimiza classifier during training
	X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=13)

	#Test data
	X_test = np.array(data_test.iloc[:, :])
	# y_test = to_categorical(np.array(data_test.iloc[:, 0]))
	# print(X_test.shape, X_train.shape)

	X = X.reshape(X.shape[0], img_rows, img_cols, 1)
	X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
	X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
	X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, 1)

	
	X = X.astype('float32')
	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')
	X_val = X_val.astype('float32')
	X /= 255
	X_train /= 255
	X_test /= 255
	X_val /= 255

	return X_train, y_train, X_val, y_val, X_test


def mConv(X, filters=8, neck=4, name='None'):
	"""
	Convolutional stack:
		- bottle neck layer to reduce the numbers of parameters
	"""
	Conv1 = Conv2D(filters, (3,3), strides=1, padding='same', activation='relu', use_bias=True, kernel_initializer=glorot_uniform(), name=name+'_in')(X)
	Conv2 = Conv2D(filters, (3,3), strides=1, padding='same', activation='relu', use_bias=True, kernel_initializer=glorot_uniform(), name=name+'_out')(Conv1)

	bottle_neck1 = Conv2D(neck, (1,1), strides=1, padding='valid', activation='relu', use_bias=True, kernel_initializer=glorot_uniform(), name=name+'_nout')(Conv2)
	return Conv2, bottle_neck1


def mnist_model(input_shape=None, num_classes=10):
	X_input = Input(input_shape, name='Input')

	X1, bn1 = mConv(X_input, 10, 5, 'Conv_1')
	X1 = MaxPooling2D(2, padding='same')(X1)
	bn1 = MaxPooling2D(2, padding='same')(bn1)

	X2, bn2 = mConv(bn1, 10, 5, 'Conv_2')

	# skip connection to allow gradient to flow more easily
	X3 = Concatenate(name='combine1')([X1, X2])
	X3 = MaxPooling2D(2, padding='same')(X3)

	bn3 = Conv2D(5, (1,1), strides=1, padding='valid', activation='relu', use_bias=True, kernel_initializer=glorot_uniform(), name='fneck')(X3)
	X = Conv2D(10, (3,3), strides=1, padding='same', activation='relu', use_bias=True, kernel_initializer=glorot_uniform(), name='final')(bn3)


	X = Flatten()(X)
	X = Dropout(0.5)(X)
	X = Dense(num_classes, activation='softmax')(X)

	model = Model(inputs=X_input, outputs=X, name='mnist')

	return model


def train(model, lr, epochs, batch_size, X, y, X_val=None, y_val=None, pre_weight=None, period=5):
	"""
		X, y: training data
		X_val, y_val: validation data
		pre_weight: path to your model's weight
		period: interval between checkpoint
	"""
	model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr),
              metrics=['accuracy'])

	model.summary()

	if pre_weight is not None:
		model.load_weights(pre_weight)

	tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

	checkpoint = keras.callbacks.ModelCheckpoint(path, monitor='acc', verbose=1, save_best_only=False, save_weights_only=True, mode='auto', period=5)

	callbacks = [tensorboard, checkpoint]

	if X_val is not None:
		model.fit(x=X, y=y,
			epochs=epochs,
			batch_size=batch_size,
			verbose=1,
			validation=(X_val, y_val),
			callbacks=callbacks)
	else:
		model.fit(x=X, y=y,
			epochs=epochs,
			batch_size=batch_size,
			verbose=1,
			callbacks=callbacks)

	return model



X_train, y_train, X_val, y_val, X_test = load_data()


model = mnist_model((28,28,1))
model = train(model, 1e-3, 0, 32, X_train, y_train)
print(model.evaluate(X_val, y_val))
# os.path.join(MODEL, 'phase_2_weights.05-0.0945-0.9715.hdf5')