import cv2, os
import numpy as np
import matplotlib.image as mpimg


IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

def load_image(data_dir, image_file):
    """
    Load RGB images from a file
    """
    image_file = str(image_file)
    return mpimg.imread(os.path.join(data_dir, image_file.strip('\'" ][')))

def resize(image):
    """
    Resize the image to the input shape used by the network model
    """
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)


def rgb2yuv(image):
    """
    Convert the image from RGB  to YUV (This is what the NVIDIA model does)
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)


def preprocess(image):
    """
    Combine all preprocess functions into one
    """
    image = resize(image)
    image = rgb2yuv(image)
    return image


def batch_generator(data_dir, image_paths, shutter_speeds, is_os, batch_size, is_training):
#    global image
    """
    Generate training image give image paths and associated camera exposure parameters.
    """
    images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    shss = np.empty(batch_size)
    isos = np.empty(batch_size)
    while True:
        i = 0
        for index in np.random.permutation(image_paths.shape[0]):
            shutter_speed = shutter_speeds[index]
            is_o = is_os[index]
            scene = image_paths[index]
            if is_training:
                image = load_image(data_dir, scene)
                images[i] = preprocess(image)
                shss[i] = shutter_speed
                isos[i] = is_o
            i += 1
            if i == batch_size:
                break
        labels = [np.array(shss), np.array(isos)]
        yield images, labels

