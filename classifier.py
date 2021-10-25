#Importing all the important models and install them if not installed on your device
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import PIL.ImageOps

#Fetching the data
X = np.load("dataset/image.npz")["arr_0"]
y = pd.read_csv("https://raw.githubusercontent.com/whitehatjr/datasets/master/C%20122-123/labels.csv")["labels"]

#Splitting the data and scaling it
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=9, train_size=3500, test_size=500)
#scaling the features
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

#Fitting the training data into the model
clf = LogisticRegression(solver='saga', multi_class='multinomial').fit(X_train_scaled, y_train)

def get_prediction(image):

    #Converting cv2 image to pil format
    im_pil = Image.open(image)

    # convert to grayscale image - 'L' format means each pixel is 
    # represented by a single value from 0 to 255
    image_bw = im_pil.convert('L')
    image_bw_resized = image_bw.resize((22,30), Image.ANTIALIAS)

    pixel_filter = 20
    min_pixel = np.percentile(image_bw_resized, pixel_filter)

    image_bw_resized_inverted_scaled = np.clip(image_bw_resized-min_pixel, 0, 255)
    max_pixel = np.max(image_bw_resized)
    image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel
    
    test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1,660)
    test_pred = clf.predict(test_sample)
    return test_pred[0]

