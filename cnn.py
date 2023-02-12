def predict(s):

 import tensorflow
 from tensorflow.keras.preprocessing.image  import ImageDataGenerator
 from tensorflow.keras.models import Sequential
 from tensorflow.keras.layers import Dense,Conv2D,Flatten,Dropout,MaxPooling2D,Activation
 from tensorflow.keras.preprocessing  import image
 import matplotlib.pyplot as plt
 import matplotlib.image as mpimg

 img_height,img_width=150,150
 train=r"C:\Users\Pavi\OneDrive\Desktop\CNN\DogsCats\DogsCats"
 test=r"C:\Users\Pavi\OneDrive\Desktop\CNN\Test\Test"
 train_sample=100
 test_sample=100
 epoch=20
 batch_size=20

 import tensorflow.keras.backend as k
 if k.image_data_format()=='channels_first':
   input_shape=(3,img_height,img_width)
 else:
   input_shape=(img_height,img_width,3)

 train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
 test_datagen=ImageDataGenerator(rescale=1./255)
 train_generator=train_datagen.flow_from_directory(train,target_size=(img_width,img_height),batch_size=batch_size,class_mode='binary',classes=['Cats','Dogs'])
 validation_generator=test_datagen.flow_from_directory(test,target_size=(img_width,img_height),batch_size=batch_size,class_mode='binary')



 model=Sequential()
 model.add(Conv2D(64,(3,3),input_shape=input_shape))
 model.add(Activation('relu'))
 model.add(MaxPooling2D(pool_size=(2,2)))
 model.add(Flatten())
 model.add(Dense(64))

 model.add(Activation('relu'))
 model.add(Dense(1))
 model.add(Activation('sigmoid'))
 model.summary()

 model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
 model.summary()

 training=model.fit_generator(train_generator,steps_per_epoch=train_sample,epochs=epoch,validation_data=validation_generator,validation_steps=test_sample)

 from tensorflow.keras.preprocessing  import image
 import numpy as np
 img_pred=image.load_img("C:\\Users\\Pavi\\OneDrive\\Desktop\\CNN\\static\\uploads\\"+s,target_size=(150,150))
 img_pred=image.img_to_array(img_pred)
 img_pred=np.expand_dims(img_pred,axis=0)
 rslt=model.predict(img_pred)

 if rslt[0][0]==1:   
   prediction="Dog"
 else:
   prediction="Cat"
 return prediction