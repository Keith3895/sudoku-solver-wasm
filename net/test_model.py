# import imageio
# import numpy as np
# from matplotlib import pyplot as plt

# im = imageio.imread("https://i.imgur.com/a3Rql9C.png")
# img_rows, img_cols = 32, 32
# gray = np.dot(im[...,:3], [0.299, 0.587, 0.114])
# plt.imshow(gray, cmap = plt.get_cmap('gray'))
# plt.show()

# gray = gray.reshape(1, img_rows, img_cols, 1)

# # normalize image
# gray /= 255

# # load the model
# from keras.models import load_model
# model = load_model("test_model.h5")

# # predict digit
# prediction = model.predict(gray)
# print(prediction.argmax())





import numpy as np
import cv2
# import pickle

########### PARAMETERS ##############
width = 640
height = 480
threshold = 0.65 # MINIMUM PROBABILITY TO CLASSIFY
cameraNo = 0
#####################################

#### CREATE CAMERA OBJECT
cap = cv2.VideoCapture(cameraNo)
cap.set(3,width)
cap.set(4,height)

#### LOAD THE TRAINNED MODEL 
# pickle_in = open("model_trained.p","rb")
# model = pickle.load(pickle_in)
from keras.models import load_model
model = load_model("test_model.h5")
#### PREPORCESSING FUNCTION
def preProcessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

while True:
    success, imgOriginal = cap.read()
    img = np.asarray(imgOriginal)
    img = cv2.resize(img,(32,32))
    img = preProcessing(img)
    cv2.imshow("Processsed Image",img)
    img = img.reshape(1,32,32,1)
    #### PREDICT
    classIndex = int(model.predict_classes(img))
    #print(classIndex)
    predictions = model.predict(img)
    #print(predictions)
    probVal= np.amax(predictions)
    print(classIndex,probVal)

    if probVal> threshold:
        cv2.putText(imgOriginal,str(classIndex) + "   "+str(probVal),
                    (50,50),cv2.FONT_HERSHEY_COMPLEX,
                    1,(0,0,255),1)

    cv2.imshow("Original Image",imgOriginal)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break