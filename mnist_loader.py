import numpy as np # linear algebra
# import skimage as ski
import struct
from array import array
from os.path  import join

#
# MNIST Data Loader Class
#
class MnistDataloader(object):
    def __init__(self, images_filepath, labels_filepath):
        self.images_filepath = images_filepath
        self.labels_filepath = labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            if i < 50: print("after reshape img: ", img)
            # img = ski.trans
            # img = img.resize((32,32), resample='nearest')
            
            images[i][:] = img    

        # print(skimage.__version__)
        # print(ski.__version__)        
        
        return images, labels
            
    def load_data(self):
        x_axis, y_axis = self.read_images_labels(self.images_filepath, self.labels_filepath)
        
        return (x_axis, y_axis)