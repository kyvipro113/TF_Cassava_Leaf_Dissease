import warnings
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

# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()  #disable for tensorFlow V2
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model

# LOAD DATA
path = '../TF_CNN_Leaf Disease/input/cassava-leaf-disease-classification/'

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

# DATA ANALYSIS

plt.figure(figsize = (18,6))
sns.countplot(y = 'label_name', data = train, order = pd.value_counts(train['label_name']).index, palette = 'muted', edgecolor = 'black')

plt.xlabel("")
plt.ylabel("")
plt.yticks(fontsize = 10)
plt.show()

train['label_name'].value_counts()
# print("DXFS")
# print(train)

# Get Image
def get_image(image_id, labels):
    
    plt.figure(figsize=(18, 8))
    
    for i, (image_id, label_name) in enumerate(zip(image_id, labels)):
        plt.subplot(4, 3, i + 1)
        image = cv2.imread(os.path.join(path, 'train_images', image_id))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        plt.imshow(image)
        plt.title(f"{label_name}", fontweight='bold', fontsize=12)
        plt.axis("off")
    
    plt.show()
# Get 12 sample per label to show
sample = train.sample(12)
image_ids = sample['image_id'].values
labels = sample['label_name'].values

get_image(image_ids, labels)

##Cassava Mosaic Disease (CMD) # B???nh kh???m l?? s???n
cmd_sample = train[train['label'] == '3'].sample(12)
image_ids = cmd_sample['image_id'].values
labels = cmd_sample['label_name'].values

get_image(image_ids, labels)

##healthy # L?? s???n b??nh th?????ng (kh???e m???nh)
healthy_sample = train[train['label'] == '4'].sample(12)
image_ids = healthy_sample['image_id'].values
labels = healthy_sample['label_name'].values

get_image(image_ids, labels)

##Cassava Green Mottle (CGM) # B???nh m???t xanh do virus secoviridae
cgm_sample = train[train['label'] == '2'].sample(12)
image_ids = cgm_sample['image_id'].values
labels = cgm_sample['label_name'].values

get_image(image_ids, labels)

##Cassava Brown Streak Disease (CBSD) # B???nh v???t n??u tr??n l?? s???n
cbsd_sample = train[train['label'] == '1'].sample(12)
image_ids = cbsd_sample['image_id'].values
labels = cbsd_sample['label_name'].values

get_image(image_ids, labels)

##Cassava Bacterial Blight (CBB) # B???nh b???c l?? s???n do vi khu???n
cbb_sample = train[train['label'] == '0'].sample(12)
image_ids = cbb_sample['image_id'].values
labels = cbb_sample['label_name'].values

get_image(image_ids, labels)

# CREATE MODEL

#Split training data and validate (test) data
train, validation = train_test_split(train, train_size = 0.8, shuffle = True, random_state = 8)

print("TRAINING")
print(train)
print("VALIDATE")
print(validation)

#MODEL using 2 dimension convulution neural netwrok 
model = tf.keras.Sequential([
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

model.compile(optimizer = RMSprop(), loss='categorical_crossentropy', metrics=['acc'])

callbacks = ReduceLROnPlateau(monitor='val_acc', 
                              factor=0.5, 
                              patience=5, 
                              verbose=1, 
                              min_lr=0.0001)

# Generate training data 
train_datagen = ImageDataGenerator(rescale=1/255, # (rescale) Chu???n h??a c??c k??nh RGB n???m trong ph???m vi 0 t???i 255 v??? 0 v?? 1
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   vertical_flip=True)

# Generate validate data
validation_datagen = ImageDataGenerator(rescale=1/255) # (rescale) Chu???n h??a c??c k??nh RGB n???m trong ph???m vi 0 t???i 255 v??? 0 v?? 1

BATCH_SIZE = 150 #256
STEPS_PER_EPOCH = train.shape[0]/BATCH_SIZE
VALIDATION_STEPS = validation.shape[0]/BATCH_SIZE
EPOCHS = 10 #20

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

history = model.fit_generator(
            train_generator,
            steps_per_epoch = STEPS_PER_EPOCH,
            epochs = EPOCHS,
            validation_data = validation_generator,
            validation_steps = VALIDATION_STEPS,
            verbose = 1,
            callbacks = [callbacks])

epochs = range(1, EPOCHS + 1)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

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

# Weights

weights = model.layers[6].get_weights()[0]
print(weights)
_weights = model.layers[6].get_weights()
print(_weights)

# Save model

model_json_save = model.to_json()
with open("model/model_save.json", "w") as json_save_file:
    json_save_file.write(model_json_save)

model.save_weights("model/model_weights.h5")

pd.DataFrame.from_dict(history.history).to_csv('model/history.csv', index=False)

print("Save model successfully!")

# Load model

json_model_save = open("model/model_save.json", "r")
load_model_json = json_model_save.read()
json_model_save.close()
Load_model = model_from_json(load_model_json)
Load_model.load_weights("model/model_weights.h5")
print("successfully!!")

# Predict
fileName = "E:/TF_CNN_Leaf Disease/input/cassava-leaf-disease-classification/test_images/2216849948.jpg"
from tensorflow.keras.preprocessing import image
import numpy as np
imgLoad = image.load_img(fileName, target_size= (150, 150))
imgLoad_arr = image.img_to_array(imgLoad)
img_batch = np.expand_dims(imgLoad_arr, axis=0)
result = model.predict(img_batch)
score = tf.nn.softmax(result[0])
print(np.argmax(score))

print("DONE")

rs = Load_model.predict(img_batch)
print(rs)

