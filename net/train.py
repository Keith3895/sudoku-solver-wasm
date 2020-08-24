import numpy as nup
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import pickle

# ------------Hoisting-------------
pathToDataset = 'DataSet'
imageDimensions = (32, 32, 3)


def listDir(path):
    lst = os.listdir(path)
    if '.DS_Store' in lst:
        lst.remove('.DS_Store')
    return lst


def drawDistribution(noOfClasses, y_train):
    print("Draw distrubutions...")
    numOfSamples = []
    for x in range(0, noOfClasses):
        # print(len(nup.where(y_train==x)[0]))
        numOfSamples.append(len(nup.where(y_train == x)[0]))
    print(numOfSamples)

    plt.figure(figsize=(10, 5))
    plt.bar(range(0, noOfClasses), numOfSamples)
    plt.title("No of Images for each Class")
    plt.xlabel("Class ID")
    plt.ylabel("Number of Images")
    plt.show()


def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img


def digitRecogModel(noOfClasses):
    noOfFilters = 60
    sizeOfFilter1 = (5, 5)
    sizeOfFilter2 = (3, 3)
    sizeOfPool = (2, 2)
    noOfNodes = 500

    model = Sequential()
    model.add((Conv2D(noOfFilters, sizeOfFilter1, input_shape=(imageDimensions[0],
                                                               imageDimensions[1], 1), activation='relu')))
    model.add((Conv2D(noOfFilters, sizeOfFilter1, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add((Conv2D(noOfFilters//2, sizeOfFilter2, activation='relu')))
    model.add((Conv2D(noOfFilters//2, sizeOfFilter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(noOfNodes, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses, activation='softmax'))

    model.compile(Adam(lr=0.001), loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def plotResult(history):
    print("Ploting model result...")
    plt.figure(1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['training', 'validation'])
    plt.title('Loss')
    plt.xlabel('epoch')
    plt.figure(2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.legend(['training', 'validation'])
    plt.title('Accuracy')
    plt.xlabel('epoch')
    plt.show()


# ------------Hoisting-------------
images = []
classNo = []
lst = listDir(pathToDataset)
numberOfClass = len(lst)
print("Total number of classes ditected", numberOfClass)
print("starting processing...")
for digitIterator in range(0, numberOfClass):
    picList = listDir(pathToDataset+"/"+str(digitIterator))
    for digitImage in picList:
        curImg = cv2.imread(pathToDataset+"/" +
                            str(digitIterator)+"/"+digitImage)
        # cv2.imshow(curImg)
        curImg = cv2.resize(curImg, (32, 32))
        images.append(curImg)
        classNo.append(digitIterator)
    print(digitIterator)
print(" ")
images = nup.array(images)
classNo = nup.array(classNo)
print("setting training set...")
X_train, X_test, Y_train, Y_test = train_test_split(
    images, classNo, test_size=0.2)
X_train, X_validation, Y_train, Y_validation = train_test_split(
    X_train, Y_train, test_size=0.2)


# drawDistribution(noOfClasses=numberOfClass, y_train=Y_train)
print("preprocessing training set...")
X_train = nup.array(list(map(preProcessing, X_train)))
X_test = nup.array(list(map(preProcessing, X_test)))
X_validation = nup.array(list(map(preProcessing, X_validation)))
print("reshaping training set...")
X_train = X_train.reshape(
    X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
X_validation = X_validation.reshape(
    X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)

print("augmenting training set...")

dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)
dataGen.fit(X_train)

print("binary matrix transform...")
Y_train = to_categorical(Y_train, numberOfClass)
Y_test = to_categorical(Y_test, numberOfClass)
Y_validation = to_categorical(Y_validation, numberOfClass)

print("Creating model...")

model = digitRecogModel(noOfClasses=numberOfClass)
print(model.summary())
print("Fit model...")
history = model.fit(dataGen.flow(X_train, Y_train,
                                           batch_size=50),
                              steps_per_epoch=131,
                              epochs=10,
                              verbose=1,
                              validation_data=(X_validation, Y_validation),
                              shuffle=1)

# plotResult(history=history)
print("running testing set...")
score = model.evaluate(X_test,Y_test,verbose=1)
print('Test Score = ',score[0])
print('Test Accuracy =', score[1])

print("saving model...")
model.save("test_model.h5",overwrite=True, include_optimizer=True)
pickle_out= open("model_trained.p", "wb")
pickle.dump(model,pickle_out)
pickle_out.close()