#parsing command line arguments
import argparse
#decoding camera images
import base64
#reading and writing files
import os
#high level file operations
import shutil
#matrix math
import numpy as np
import cv2
#image manipulation
from PIL import Image
from keras.models import load_model
#helper class
import utils
#os.environ['CUDA_VISIBLE_DEVICES'] = '7'

#init our model and image array as empty
model = None
prev_image_array = None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Autonomus shooting')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
        args = parser.parse_args()
   

image = Image.open("frame.jpg")
try:
    image1 = np.asarray(image)  # from PIL image to numpy array
    image1 = utils.preprocess(image1)  # apply the preprocessing
    image1 = np.array([image1])  # the model expects 4D array
    model = load_model('model-004.h5')
    model.summary()
    shs, iso = float(model.predict(image1, batch_size=1))
    print('Shutter Speed={} ISO={}'.format(shs ,iso))










