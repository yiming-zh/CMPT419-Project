"""
    SFU CMPT 419/726 Research
"""
import os
import numpy as np
from PIL import Image

PATH = "dataset/"

IMAGE_WIDTH = 50
IMAGE_HEIGHT = 50


class DataLoader:
    def __init__(self):
        print("\nPlease wait, loading data...")
        self.__test_files = os.listdir("%stest/" % PATH)
        self.__train_files = os.listdir("%strain/" % PATH)

    def get_data(self):
        train_files = self.__train_files
        return {
            "image": self.read_img(train_files, 'train'),
            "glass_labels": self.read_label(train_files, 0),
            "gender_labels": self.read_label(train_files, 1),
        }

    def get_test_data(self):
        test_files = self.__test_files[:16]
        return {
            "image": self.read_img(test_files, 'test'),
            "names": test_files
        }

    def read_img(self, files, data_type):
        arr = []
        for file in files:
            img = Image.open("%s/%s/%s" % (PATH, data_type, file))
            data = img.load()
            view = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype=np.float)
            for i in range(IMAGE_HEIGHT):
                for j in range(IMAGE_WIDTH):
                    r, g, b = data[j, i]
                    view[i, j, 0] = (r + g + b) // 3
            arr.append(view)
        return np.array(arr)

    def read_label(self, files, number):
        glass_arr = []
        gender_arr = []
        for file in files:
            glass_label = int(file[0])
            glass_view = np.zeros(2, dtype=np.float)
            glass_view[glass_label] = 1
            glass_arr.append(glass_view)
            gender_label = int(file[2])
            gender_view = np.zeros(2, dtype=np.float)
            gender_view[gender_label] = 1
            gender_arr.append(gender_view)
        if number == 1:
            return np.array(glass_arr)
        else:
            return np.array(gender_arr)
