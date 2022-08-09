from typing import List

from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, plot_confusion_matrix

trainPath = 'Faces/Small/Train'
validPath = 'Faces/Small/Valid'

trainBatches = ImageDataGenerator(
    preprocessing_function=keras.applications.mobilenet.preprocess_input).flow_from_directory(trainPath,
                                                                                              target_size=(224, 224),
                                                                                              color_mode='rgb',
                                                                                              batch_size=10)
validBatches = ImageDataGenerator(
    preprocessing_function=keras.applications.mobilenet.preprocess_input).flow_from_directory(validPath,
                                                                                              target_size=(224, 224),
                                                                                              color_mode='rgb',
                                                                                              batch_size=10)


mobile = keras.applications.mobilenet.MobileNet()
print(mobile.summary())
x = mobile.layers[-6].output
predictions = Dense(2, activation="softmax")(x)
model = Model(inputs=mobile.input, outputs=predictions)
for layer in model.layers[:-33]:
    layer.trainable = False

model.compile(Adam(lr=.00005), loss=categorical_crossentropy, metrics=['accuracy'])
mc = ModelCheckpoint('balanced2.hdf5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
callbackslist: List[ModelCheckpoint] = [mc]
model.fit(trainBatches, steps_per_epoch=20, validation_data=validBatches, validation_steps=3, epochs=100,
                    callbacks=callbackslist, verbose=2, )
