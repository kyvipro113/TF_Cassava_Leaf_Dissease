import warnings

from scipy.sparse import data
warnings.filterwarnings("ignore")

import os
import json
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import cv2

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf

from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

class Model_CNN_Classfication(object):
    def __iniit__(self):
        self.model = None
        self.history = None
        self.acc = None
        self.val_acc = None
        self.loss = None
        self.val_loss = None
        self.Load_model = None

    # CREATE MODEL
    def create_Model(self):
        #MODEL using 2 dimension convulution neural netwrok 
        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (5, 5), activation='relu', input_shape=(150, 150, 3)),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (5, 5), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(128, (5, 5), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(128, (5, 5), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(5, activation='softmax')
        ])

        self.model.compile(optimizer = RMSprop(), loss='categorical_crossentropy', metrics=['acc'])

        self.callbacks = ReduceLROnPlateau(monitor='val_acc', 
                              factor=0.5, 
                              patience=5, 
                              verbose=1, 
                              min_lr=0.0001)
    # TRAINING MODEL
    def training_Model(self, BatchSize, Epochs):
        # LOAD DATA
        #path = '../TF_CNN_Leaf Disease/input/cassava-leaf-disease-classification/'
        path = '../TF_CNN_Leaf Disease/input/cassava_leaf_disease_classfication_normal/'
        os.listdir(path)
        print('No of Train images: ' + str(len(os.listdir(path + 'train_images'))))
        print('No of Test images: ' + str(len(os.listdir(path + 'test_images'))))

        train = pd.read_csv(path + 'train.csv', sep=',')
        train.head()

        print("TOTAL DATA")
        print(train)
        print("TOTAL DATA: {}".format(train.shape))
        print("\n")

        with open(os.path.join(path + 'label_num_to_disease_map.json')) as f:
            label_name = json.loads(f.read())  

        print(json.dumps(label_name, indent = 1))
        print("\n")     

        train['label'] = train['label'].astype(str)
        train['label_name'] = train['label'].map(label_name)
        train.head()

        print("TOTAL DATA AND LABEL'S NAME")
        print(train)
        print("TOTAL DATA  TRAIN AND LABEL'S NAME: {}".format(train.shape))
        print("\n")

        #Split training data and validate (test) data
        train, validation = train_test_split(train, train_size = 0.8, shuffle = True, random_state = 8)

        print("TRAINING")
        print(train)
        print("VALIDATE")
        print(validation)

        # Generate training data 
        train_datagen = ImageDataGenerator(rescale=1/255, # (rescale) Chuẩn hóa các kênh RGB nằm trong phạm vi 0 tới 255 về 0 tới 1
                                        rotation_range=40,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        shear_range=0.2,
                                        zoom_range=0.2,
                                        horizontal_flip=True,
                                        vertical_flip=True)

        # Generate validate data
        validation_datagen = ImageDataGenerator(rescale=1/255) # (rescale) Chuẩn hóa các kênh RGB nằm trong phạm vi 0 tới 255 về 0 tới 1

        BATCH_SIZE = BatchSize #256
        STEPS_PER_EPOCH = train.shape[0]/BATCH_SIZE # Interator
        VALIDATION_STEPS = validation.shape[0]/BATCH_SIZE
        EPOCHS = Epochs #20

        train_generator = train_datagen.flow_from_dataframe(train, 
                                                    directory = os.path.join(path, 'train_images'),
                                                    x_col = 'image_id',
                                                    y_col = 'label',
                                                    target_size = (150, 150),
                                                    batch_size = BATCH_SIZE,
                                                    class_mode = 'categorical')

        validation_generator = validation_datagen.flow_from_dataframe(validation, 
                                                    directory = os.path.join(path, 'train_images'),
                                                    x_col = 'image_id',
                                                    y_col = 'label',
                                                    target_size = (150,150),
                                                    batch_size = BATCH_SIZE,
                                                    class_mode = 'categorical')
        
        self.history = self.model.fit_generator(
            train_generator,
            steps_per_epoch = STEPS_PER_EPOCH,
            epochs = EPOCHS,
            validation_data = validation_generator,
            validation_steps = VALIDATION_STEPS,
            verbose = 1,
            callbacks = [self.callbacks])

        self.epochs = range(1, EPOCHS + 1)

        self.acc = self.history.history['acc']
        self.val_acc = self.history.history['val_acc']
        self.loss = self.history.history['loss']
        self.val_loss = self.history.history['val_loss']
    # SAVE MODEL
    def save_model(self):
        # Save model to json file
        model_json_save = self.model.to_json()
        with open("model/model_save.json", "w") as json_save_file:
            json_save_file.write(model_json_save)
        # Save weights
        self.model.save_weights("model/model_weights.h5")
        # Save history
        pd.DataFrame.from_dict(self.history.history).to_csv('model/history.csv', index=False)

        print("SAVE MODEL SUCCESSFULLY!")
    # LOAD MODEL
    def load_model(self):
        # Open json file and load model
        json_model_save = open("model/model_save.json", "r")
        load_model_json = json_model_save.read()
        json_model_save.close()
        self.Load_model = model_from_json(load_model_json)
        # Load weights
        self.Load_model.load_weights("model/model_weights.h5")
        print("LOAD MODEL SUCCESSFULLY!!")
        return self.Load_model
    # PREDICT
    def predict(self, filePath):
        fileName = filePath
        imgLoad = image.load_img(fileName, target_size= (150, 150))
        imgLoad_arr = image.img_to_array(imgLoad)
        img_batch = np.expand_dims(imgLoad_arr, axis=0)
        result = self.Load_model.predict(img_batch)
        print(result[0])
        #score = tf.nn.softmax(result[0])
        score = np.argmax(result)
        print(score)
        #print(np.argmax(score))
        ID_Label = np.argmax(score)
        print("DONE")
        return ID_Label

if __name__ == "__main__":
    MODEL = Model_CNN_Classfication()
    #MODEL.create_Model()
    #MODEL.training_Model(BatchSize=2, Epochs=3)
    # L_MODEL = MODEL.load_model()
    # ID_LB = MODEL.predict(filePath="E:/TF_CNN_Leaf Disease/input/cassava-leaf-disease-classification/test_images/2216849948.jpg")
    # print(ID_LB)

    data_history = pd.read_csv('model/history.csv', sep=',')
    data_history.head
    AE = data_history.shape[0]
    print(data_history)
    epochs = []
    for i in range(AE):
        epochs.append(i)
    epochs = np.array(epochs)
    print(epochs.shape)
    print(epochs)
   

    acc = data_history['acc']
    val_acc = data_history['val_acc']
    loss = data_history['loss']
    val_loss = data_history['val_loss']

    print(acc)
    print(val_acc)
    print(val_acc.shape)
    print(type(acc))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    ax1.plot(epochs, acc, label = 'Training Accuracy')
    ax1.plot(epochs, val_acc, label = 'Validation Accuracy')
    ax1.set_title('Training & Validation Accuracy', fontweight='bold', fontsize=10)
    ax1.legend()

    ax2.plot(epochs, loss, label = 'Training loss')
    ax2.plot(epochs, val_loss, label = 'Validation loss')
    ax2.set_title('Training & Validation Loss', fontweight='bold', fontsize=10)
    ax2.legend()

    plt.show()
        

    