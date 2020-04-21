"""
    SFU CMPT 419/726 Research
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from data_loader import DataLoader
from keras.callbacks import TensorBoard

EPOCHS = 30
LEARNING_RATE = 0.001
BATCH_SIZE = 32
IMAGE_WIDTH = 50
IMAGE_HEIGHT = 50


class MultiCNN:
    def __init__(self):
        self.input = keras.layers.Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 1))

    def glass_classifier(self, input):
        result = keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(input)
        result = keras.layers.MaxPool2D(pool_size=(2, 2))(result)
        result = keras.layers.Flatten()(result)
        result = keras.layers.Dense(64, activation=tf.nn.relu)(result)
        result = keras.layers.Dropout(0.5)(result)
        result = keras.layers.Dense(2, activation=tf.nn.softmax, name="glass_result")(result)
        return result

    def gender_classifier(self, input):
        result = keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(input)
        result = keras.layers.MaxPool2D(pool_size=(2, 2))(result)
        result = keras.layers.Flatten()(result)
        result = keras.layers.Dense(128, activation=tf.nn.relu)(result)
        result = keras.layers.Dropout(0.35)(result)
        result = keras.layers.Dense(2, activation=tf.nn.softmax, name="gender_result")(result)
        return result

    def build_network(self):
        result = keras.layers.Conv2D(32, (3, 3), strides=(1, 1), activation='relu', name="common_conv")(self.input)
        result = keras.layers.MaxPool2D(pool_size=(2, 2))(result)
        glasses = self.glass_classifier(result)
        gender = self.gender_classifier(result)
        model = keras.models.Model(inputs=self.input, outputs=[glasses, gender], name="multi_cnn")
        return model


def train_network():
    data = DataLoader()
    train_data = data.get_data()
    train_x = train_data['image'] / 255
    glass_train_y = train_data['glass_labels']
    gender_train_y = train_data['gender_labels']
    multi_cnn = MultiCNN()
    callback = [tf.keras.callbacks.EarlyStopping(patience=2, min_delta=0, monitor='val_loss'), TensorBoard(log_dir='logs')]
    model = multi_cnn.build_network()
    model.summary()
    losses = {'glass_result': 'categorical_crossentropy', 'gender_result': 'categorical_crossentropy'}
    loss_weights = {'glass_result': 1.0, 'gender_result': 1.0}
    optimizer = keras.optimizers.Adam(lr=LEARNING_RATE, decay=LEARNING_RATE / EPOCHS)
    model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=["accuracy"])
    model.fit(train_x, {"glass_result": glass_train_y, "gender_result": gender_train_y},
              epochs=EPOCHS, verbose=1, validation_split=0.1, shuffle=True, callbacks=callback)
    choice = input("\nFinish training. Do you want to save the trained model? [y/n]:")
    if choice == 'y':
        model.save("cnn_model.h5")
        print("\nModel saved.")


def test_network():
    model = tf.keras.models.load_model("cnn_model.h5")
    data = DataLoader()
    test = data.get_test_data()["image"]
    prediction = model.predict(test)
    plt.figure(figsize=(10, 7))
    for i in range(16):
        glass_result = prediction[0][i]
        gender_result = prediction[1][i]
        glass_max_label = int(np.argmax(glass_result))
        gender_max_label = int(np.argmax(gender_result))
        plt.subplot(4, 8, 2 * i + 1)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        image = test.reshape(test.shape[0], IMAGE_WIDTH, IMAGE_HEIGHT)[i]
        plt.imshow(image, cmap="gray")
        plt.xlabel("{}\n{}".format(["Female", "Male"][gender_max_label], ["No glasses", "With glasses"][glass_max_label]))
    plt.show()

    # Show 1st conv layer
    # intermediate_layer_model = keras.models.Model(inputs=model.input, outputs=model.get_layer("common_conv").output)
    # intermediate_output = intermediate_layer_model.predict(test)[0]
    # temp = []
    # for j in range(4):
    #     temp.append(np.concatenate([intermediate_output[:, :, i+(j*8)] for i in range(8)], axis=1))
    # temp = np.concatenate([temp[i] for i in range(4)], axis=0)
    # plt.figure(figsize=(10, 5))
    # plt.imshow(temp)
    # plt.show()


if __name__ == '__main__':
    print("\n1.Train a new network")
    print("2.Test a pre-trained network (saved as cnn_model.h5)")
    choice = int(input("\nPlease enter 1 or 2: "))
    if choice == 1:
        train_network()
    elif choice == 2:
        test_network()
    else:
        print("Invalid Input!")
