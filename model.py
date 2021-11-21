import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def get_model():

	img_input = layers.Input(shape=(48,64,2))

	conv1 = layers.Conv2D(32, 3, strides=1, padding="same")(img_input)
	bn1 = layers.BatchNormalization()(conv1)
	lr1 = layers.LeakyReLU(0.2)(bn1)
	conv2 = layers.Conv2D(32, 3, strides=2, padding="same")(lr1)
	bn2 = layers.BatchNormalization()(conv2)
	lr2 = layers.LeakyReLU(0.2)(bn2)
	#(24,32)

	conv3 = layers.Conv2D(64, 3, strides=1, padding="same")(lr2)
	bn3 = layers.BatchNormalization()(conv3)
	lr3 = layers.LeakyReLU(0.2)(bn3)
	conv4 = layers.Conv2D(64, 3, strides=2, padding="same")(lr3)
	bn4 = layers.BatchNormalization()(conv4)
	lr4 = layers.LeakyReLU(0.2)(bn4)
	#(12,16)

	conv5 = layers.Conv2D(128, 3, strides=1, padding="same")(lr4)
	bn5 = layers.BatchNormalization()(conv5)
	lr5 = layers.LeakyReLU(0.2)(bn5)
	conv6 = layers.Conv2D(128, 3, strides=2, padding="same")(lr5)
	bn6 = layers.BatchNormalization()(conv6)
	lr6 = layers.LeakyReLU(0.2)(bn6)
	#(6,8)

	latent = layers.Flatten()(lr6)
	class_pred = layers.Dense(1)(latent)
	class_pred = layers.Activation('sigmoid')(class_pred)
	#class_pred = layers.Softmax()(class_pred)

	up7 = layers.UpSampling2D((2,2),interpolation="nearest")(lr6)
	concat7 = layers.Concatenate(axis=-1)([up7,lr5])
	conv7 = layers.Conv2D(128, 3, strides=1, padding="same")(concat7)
	bn7 = layers.BatchNormalization()(conv7)
	lr7 = layers.LeakyReLU(0.2)(bn7)
	conv8 = layers.Conv2D(128, 3, strides=1, padding="same")(lr7)
	bn8 = layers.BatchNormalization()(conv8)
	lr8 = layers.LeakyReLU(0.2)(bn8)
	#(12,16)

	up9 = layers.UpSampling2D((2,2),interpolation="nearest")(lr8)
	concat9 = layers.Concatenate(axis=-1)([up9,lr3])
	conv9 = layers.Conv2D(64, 3, strides=1, padding="same")(concat9)
	bn9 = layers.BatchNormalization()(conv9)
	lr9 = layers.LeakyReLU(0.2)(bn9)
	conv10 = layers.Conv2D(64, 3, strides=1, padding="same")(lr9)
	bn10 = layers.BatchNormalization()(conv10)
	lr10 = layers.LeakyReLU(0.2)(bn10)
	#(24,32)

	up11 = layers.UpSampling2D((2,2),interpolation="nearest")(lr10)
	concat11 = layers.Concatenate(axis=-1)([up11,lr1])
	conv11 = layers.Conv2D(32, 3, strides=1, padding="same")(concat11)
	bn11 = layers.BatchNormalization()(conv11)
	lr11 = layers.LeakyReLU(0.2)(bn11)
	conv12 = layers.Conv2D(32, 3, strides=1, padding="same")(lr11)
	bn12 = layers.BatchNormalization()(conv12)
	lr12 = layers.LeakyReLU(0.2)(bn12)
	#(34,64)???????

	img_output = layers.Conv2D(1, 3, strides=1, padding="same")(lr12)
	img_output = layers.Activation('tanh')(img_output)

	model = keras.models.Model(inputs=img_input, outputs=[class_pred,img_output], name="HELLO")
	return(model)

	