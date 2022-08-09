#imports necessary modules
import cv2
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import Reshape
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
import numpy as np

#specifies path where test images are stored
testPath = 'Faces/Small/TestImages'
#loads images in batches according to their classes
testBatches = ImageDataGenerator(
    preprocessing_function=keras.applications.mobilenet.preprocess_input).flow_from_directory(testPath,
                                                                                              target_size=(224, 224),
                                                                                              color_mode='rgb',
                                                                                              batch_size=10,
                                                                                              shuffle=False)


#procedure to plot a confusion matrix of predictions
def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

#loads in the model
m=keras.models.load_model('ACCURATEMODEL.hdf5')

#creates labels for test images according to their classes
test_labels=testBatches.classes
#runs predictions on test images using model and its predict method, stores results in predictions
predictions=m.predict(x=testBatches,batch_size=17,verbose=1)
print(predictions)
#gets the max argument from predictions and rounds it
rounded_predictions=np.argmax(predictions, axis=-1)
print(rounded_predictions)
#creates confusion matrix with predictions and true classes
cm = confusion_matrix(y_true=test_labels, y_pred=rounded_predictions)
labels=['Not Drowsy','Drowsy']
#plots the confusion matrix
plot_confusion_matrix(cm,labels,title='CM')
