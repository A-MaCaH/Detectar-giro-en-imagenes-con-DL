from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D

num_classes = 2
resnet_weights_path = '/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

my_new_model = Sequential()
my_new_model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))
my_new_model.add(Dense(num_classes, activation='softmax'))

# Indicate whether the first layer should be trained/changed or not.
my_new_model.layers[0].trainable = False

my_new_model.compile(optimizer='sgd', 
                     loss='categorical_crossentropy', 
                     metrics=['accuracy'])

from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_size = 224
data_generator = ImageDataGenerator()


train_generator = data_generator.flow_from_directory(
                                        directory=r"C:\Users\THE EYE\Desktop\AlgoIA\CNN\transferLearning\images\train",
                                        target_size=(image_size, image_size),
                                        batch_size=10,
                                        class_mode='categorical')

validation_generator = data_generator.flow_from_directory(
                                        directory=r"C:\Users\THE EYE\Desktop\AlgoIA\CNN\transferLearning\images\val",
                                        target_size=(image_size, image_size),
                                        class_mode='categorical')

# fit_stats below saves some statistics describing how model fitting went
# the key role of the following line is how it changes my_new_model by fitting to data
fit_stats = my_new_model.fit_generator(train_generator,
                                       steps_per_epoch=22,
                                       validation_data=validation_generator,
                                       validation_steps=1)

my_new_model.save('classifier.h5')

from keras.preprocessing import image
import numpy as np
test_image = image.load_img(r"C:\Users\THE EYE\Desktop\AlgoIA\CNN\transferLearning\test\imgTest.jpg", target_size = (image_size, image_size)) 
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

test_image1 = image.load_img(r"C:\Users\THE EYE\Desktop\AlgoIA\CNN\transferLearning\test\imgTest1.jpg", target_size = (image_size, image_size)) 
test_image1 = image.img_to_array(test_image1)
test_image1 = np.expand_dims(test_image1, axis = 0)

test_image2 = image.load_img(r"C:\Users\THE EYE\Desktop\AlgoIA\CNN\transferLearning\test\imgTest2.jpg", target_size = (image_size, image_size)) 
test_image2 = image.img_to_array(test_image2)
test_image2 = np.expand_dims(test_image2, axis = 0)
#predict the result
result = my_new_model.predict(test_image)
result1 = my_new_model.predict(test_image1)
result2 = my_new_model.predict(test_image2)

print(result)
print(result1)
print(result2)

print(my_new_model.predict_classes(test_image))
print(my_new_model.predict_classes(test_image1))
print(my_new_model.predict_classes(test_image2))


