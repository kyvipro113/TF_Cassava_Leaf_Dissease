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
#path = '../TF_CNN_Leaf Disease/input/cassava-leaf-disease-classification/'
path = '../TF_CNN_Leaf Disease/input/cassava_leaf_disease_classfication_normal/'

os.listdir(path)
print('No of Train images: ' + str(len(os.listdir(path + 'train_images'))))
print('No of Test images: ' + str(len(os.listdir(path + 'test_images'))))

train = pd.read_csv(path + 'train.csv', sep=',')
train.drop(train.iloc[:, -1:1], axis=1)
train.head()

print("============TOTAL DATA============")
print(train)
print("TOTAL DATA: {}".format(train.shape))
print("\n")

df = train['label']
print(df)

with open(os.path.join(path + 'label_num_to_disease_map.json')) as f:
    label_name = json.loads(f.read())
print("="*24)    
print(json.dumps(label_name, indent = 1))
print("\n")

train['label'] = train['label'].astype(str)
train['label_name'] = train['label'].map(label_name)
train.head()

print("========TOTAL DATA AND LABEL'S NAME========")
print(train)
print("="*24)
CMD = train[['image_id', 'label']]

print("="*24)
print("TOTAL DATA  TRAIN AND LABEL'S NAME: {}".format(train.shape))
print("\n")

# # DATA ANALYSIS

plt.figure(figsize = (18,6))
sns.countplot(y = 'label_name', data = train, order = pd.value_counts(train['label_name']).index, palette = 'muted', edgecolor = 'black')

plt.xlabel("")
plt.ylabel("")
plt.yticks(fontsize = 10)
plt.show()

train['label_name'].value_counts()
# # print("DXFS")
# # print(train)

# # Get Image
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
# # Get 12 sample per label to show
sample = train.sample(12)
image_ids = sample['image_id'].values
labels = sample['label_name'].values

get_image(image_ids, labels)

##Cassava Mosaic Disease (CMD) # Bệnh khảm lá sắn
cmd_sample = train[train['label'] == '3'].sample(12)
image_ids = cmd_sample['image_id'].values
labels = cmd_sample['label_name'].values

get_image(image_ids, labels)

##healthy # Lá sắn bình thường (khỏe mạnh)
healthy_sample = train[train['label'] == '4'].sample(12)
image_ids = healthy_sample['image_id'].values
labels = healthy_sample['label_name'].values

get_image(image_ids, labels)

##Cassava Green Mottle (CGM) # Bệnh mọt xanh do virus secoviridae
cgm_sample = train[train['label'] == '2'].sample(12)
image_ids = cgm_sample['image_id'].values
labels = cgm_sample['label_name'].values

get_image(image_ids, labels)

# ##Cassava Brown Streak Disease (CBSD) # Bệnh vệt nâu trên lá sắn
cbsd_sample = train[train['label'] == '1'].sample(12)
image_ids = cbsd_sample['image_id'].values
labels = cbsd_sample['label_name'].values

get_image(image_ids, labels)

# ##Cassava Bacterial Blight (CBB) # Bệnh bạc lá sắn do vi khuẩn
cbb_sample = train[train['label'] == '0'].sample(12)
image_ids = cbb_sample['image_id'].values
labels = cbb_sample['label_name'].values

get_image(image_ids, labels)

# #Split training data and validate (test) data
train, validation = train_test_split(train, train_size = 0.8, shuffle = True, random_state = 8)

print("TRAINING")
print(train)
print("VALIDATE")
print(validation)
