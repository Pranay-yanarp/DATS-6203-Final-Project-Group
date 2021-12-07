import tensorflow
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import cv2
from skimage.transform import rotate
from skimage.util import random_noise
from skimage.exposure import adjust_gamma
from skimage.filters import gaussian
import numpy as np
import glob
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report

from tensorflow.python.framework.ops import tensor_id


# re-size all the images to this
IMAGE_SIZE = [224, 224]

train_path = '/train'
valid_path = '/val'
classes = glob.glob('train/*')

#Data Loader
train_datagen = ImageDataGenerator()
val_datagen = ImageDataGenerator()
training_set = train_datagen.flow_from_directory('train',
                                                 target_size = (224, 224),
                                                 batch_size = 64,
                                                 class_mode = 'categorical')
val_set = val_datagen.flow_from_directory('val',
                                            target_size = (224, 224),
                                            batch_size = 64,
                                            class_mode = 'categorical')



model = DenseNet121(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
        for layer in model.layers:
            layer.trainable = False
        x = Flatten()(model.output)
        prediction = Dense(len(classes), activation='softmax')(x)
        model = Model(inputs=model.input, outputs=prediction)
        model.compile(
            loss='categorical_crossentropy',
            optimizer='Adam',
            metrics=['accuracy']
        )


check_point = tensorflow.keras.callbacks.ModelCheckpoint('model_densenet121.h5', monitor='accuracy', save_best_only=True)

m = model.fit(
  training_set,
  validation_data=val_set,
  epochs=30,
  steps_per_epoch=len(training_set),
  validation_steps=len(val_set),
  callbacks=[check_point]
)

val_preds = final_model.predict(np.array([x for i in range(len(val_set)) for x in val_set[i][0]]))
val_preds = np.argmax(val_preds, axis=1)
val_actual = [x for i in range(len(val_set)) for x in val_set[i][1]]
val_actual = np.argmax(val_actual, axis=1)

print("Accuracy:" + str(accuracy_score(val_actual, val_preds)))
print("F1 Score:" + str(f1_score(val_actual, val_preds, average='micro')))
print("Confusion Matrix:\n" + str(confusion_matrix(val_actual, val_preds)))
print("Classification Report:\n" + str(classification_report(val_actual, val_preds)))


#Test Model
final_model = tensorflow.keras.models.load_model('model_resnet50.h5')

test_image = image.load_img('test/diseased cotton leaf/dis_leaf (322).jpg', target_size = (224, 224))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
preds = final_model.predict(test_image)
preds = np.argmax(preds, axis=1)

if preds==0:
  print("The leaf is diseased cotton leaf")
elif preds==1:
  print("The leaf is diseased cotton plant")
elif preds==2:
  print("The leaf is fresh cotton leaf")
else:
  print("The leaf is fresh cotton plant")



# =================== application================================



from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
import tensorflow as tf

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# Keras
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH ='model_resnet50.h5'

# Load your trained model
model = load_model(MODEL_PATH)




def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    #x=x/255
    x = np.expand_dims(x, axis=0)
   

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
   # x = preprocess_input(x)

    preds = model.predict(x)
    preds=np.argmax(preds, axis=1)
    print(preds)
    if preds==0:
        preds="The leaf is diseased cotton leaf"
    elif preds==1:
        preds="The leaf is diseased cotton plant"
    elif preds==2:
        preds="The leaf is fresh cotton leaf"
    else:
        preds="The leaf is fresh cotton plant"
        
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True,port=9989,use_reloader=False,threaded=False)
