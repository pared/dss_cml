import json
from keras.datasets import mnist
from keras.models import Sequential 
from keras.layers.core import Dense, Activation
from keras.utils import np_utils

import numpy as np     
np.random.seed(0) 

def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
 
    x_train = x_train.reshape(60000, 784)     
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')     
    x_test = x_test.astype('float32')     
    x_train /= 255    
    x_test /= 255
    classes = 10
    y_train = np_utils.to_categorical(y_train, classes)     
    y_test = np_utils.to_categorical(y_test, classes)
    return (x_train, y_train), (x_test, y_test)
 
def get_model():
    model = Sequential()     
    model.add(Dense(512, input_dim=784)) 
    model.add(Activation('relu'))     
    model.add(Dense(512, input_dim=512)) 
    model.add(Activation('relu'))     
    model.add(Dense(512, input_dim=512)) 
    model.add(Activation('sigmoid'))     
    model.add(Dense(512, input_dim=512)) 
    model.add(Activation('relu'))     
    model.add(Dense(512, input_dim=512)) 
    model.add(Activation('relu'))     

    model.add(Dense(10, input_dim=512)) 

    model.add(Activation('softmax'))
 
    model.compile(loss='categorical_crossentropy', 
    metrics=['accuracy'], optimizer='sgd')
    return model

def write_confusion_matrix(y_test, y_pred, filename="confusion_matrix.png"):
    cls_pred = y_pred.argmax(axis=-1)
    cls_test = y_test.argmax(axis=-1)

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(cls_test, cls_pred)
    import matplotlib.pyplot as plt
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.xticks([], [])
    plt.yticks([], [])
    plt.title('Confusion matrix')
    plt.savefig(filename)
    

def write_confusion_matrix_csv(y_test, y_pred, filename="confusion_matrix.csv"):
    cls_pred = y_pred.argmax(axis=-1)
    cls_test = y_test.argmax(axis=-1)

    with open(filename, "w") as fd:
        fd.write("actual,predicted\n")
        for i in range(len(cls_pred)):
            fd.write(f"{cls_test[i]},{cls_pred[i]}\n")

(x_train, y_train), (x_test, y_test) = load_data()
model = get_model()

model.fit(x_train, y_train, batch_size=128, epochs=1)
score = model.evaluate(x_test, y_test)

y_pred = model.predict(x_test)
write_confusion_matrix(y_test, y_pred)
write_confusion_matrix_csv(y_test, y_pred) 

accuracy = score[1]
with open("metrics.json", "w") as fd:
    json.dump({'accuracy' : accuracy}, fd)
