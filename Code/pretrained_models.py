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

'''
files = glob.glob('train/*/*.jpg', recursive = True)
for i in files:
    aug = cv2.imread(i)
    aug1 = adjust_gamma(aug, gamma=0.5,gain=1).astype(np.uint8)
    cv2.imwrite('train\\' + i.split('\\')[1] + '\\aug1_' + i.split('\\')[2], aug1)

    aug1 = adjust_gamma(aug, gamma=2, gain=1).astype(np.uint8)
    cv2.imwrite('train\\' + i.split('\\')[1] + '\\aug2_' + i.split('\\')[2], aug1)

    aug1 = np.fliplr(aug)
    cv2.imwrite('train\\' + i.split('\\')[1] + '\\aug3_' + i.split('\\')[2], aug1)

    aug1 = np.flipud(aug)
    cv2.imwrite('train\\' + i.split('\\')[1] + '\\aug4_' + i.split('\\')[2], aug1)

    aug1 = (random_noise(aug)*255).astype(np.uint8)
    cv2.imwrite('train\\' + i.split('\\')[1] + '\\aug5_' + i.split('\\')[2], aug1)
'''
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

def initialize_model(model_name, classes, IMAGE_SIZE):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.

    if model_name == "resnet":
        """ Resnet50
        """
        model = ResNet50(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
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
        check_point = tensorflow.keras.callbacks.ModelCheckpoint('model_resnet50.h5', monitor='accuracy', save_best_only=True)

    elif model_name == "vgg":
        """ vgg16
        """
        model = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
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
        check_point = tensorflow.keras.callbacks.ModelCheckpoint('model_vgg16.h5', monitor='accuracy',save_best_only=True)

    elif model_name == "densenet":
        """ Densenet121
        """
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

    else:
        print("Invalid model name, exiting...")
        exit()

    return model, check_point


model, check_point = initialize_model('vgg', classes, IMAGE_SIZE)

m = model.fit(
  training_set,
  validation_data=val_set,
  epochs=30,
  steps_per_epoch=len(training_set),
  validation_steps=len(val_set),
  callbacks=[check_point]
)

#plots
# summarize history for accuracy
plt.plot(m.history['accuracy'])
plt.plot(m.history['val_accuracy'])
plt.title('VGG16 model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(m.history['loss'])
plt.plot(m.history['val_loss'])
plt.title('VGG16 model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()




val_preds = final_model.predict(np.array([x for i in range(len(val_set)) for x in val_set[i][0]]))
val_preds = np.argmax(val_preds, axis=1)
val_actual = [x for i in range(len(val_set)) for x in val_set[i][1]]
val_actual = np.argmax(val_actual, axis=1)

print("Accuracy:" + str(accuracy_score(val_actual, val_preds)))
print("F1 Score:" + str(f1_score(val_actual, val_preds, average='micro')))
print("Confusion Matrix:\n" + str(confusion_matrix(val_actual, val_preds)))
print("Classification Report:\n" + str(classification_report(val_actual, val_preds)))


#Test Model
final_model = tensorflow.keras.models.load_model('model_densenet121.h5')

test_image = image.load_img('test/diseased cotton leaf/dis_leaf (322).jpg', target_size = (224, 224))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
preds = final_model.predict(test_image)
preds = np.argmax(preds, axis=1)

if preds==0:
  print("Diseased cotton leaf")
elif preds==1:
  print("Diseased cotton plant")
elif preds==2:
  print("Fresh cotton leaf")
else:
  print("Fresh cotton plant")

'''
References:-
https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50/ResNet50
https://www.tensorflow.org/api_docs/python/tf/keras/applications/densenet/DenseNet121
https://www.tensorflow.org/api_docs/python/tf/keras/applications/vgg16/VGG16
https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
'''