Different TF2 based Pretrained Model which generate feature vector are availaible at 
https://tfhub.dev/s?module-type=image-feature-vector&tf-version=tf2




***

---
___

## MTCNN 

[What is MTCNN](https://towardsdatascience.com/how-does-a-face-detection-program-work-using-neural-networks-17896df8e6ff)

[Structure of MTCNN](https://towardsdatascience.com/face-detection-neural-network-structure-257b8f6f85d1)




    multi task Cascaded convolution network

    https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html

this basically return us the boundary boxes in which the face was found. this make the accuracy of the later facenet to be much higher. [Implementation Here](https://github.com/davidsandberg/facenet/blob/master/src/align/align_dataset_mtcnn.py)


## FaceNet 

## MTCNN + Facenet 
https://medium.com/@mohitsaini_54300/train-facenet-with-triplet-loss-for-real-time-face-recognition-a39e2f4472c3



could be used with

1. YOLO -> To predict if there is an type availaible in the images  
2. YOLO -> To predict if there is an type availaible in the images
mobile-net -- > to detect the object availaible in the image large category output in this 
Emojo scavanger uses this to re-train their limited net for 80 images reduced from the original 400 classes

[Tensor Flow Blog](https://blog.tensorflow.org/2018/10/)

[Emoji Scavneger](how-we-built-emoji-scavenger-hunt-using-tensorflow-js.html)



## How transfer learning works :

> This shows an existing model loaded as a layer to create a new one after adding a new `FC layer `

https://www.tensorflow.org/hub/tutorials/tf2_image_retraining

In short and crisp summary of all above is :
    
    load the model in keras and feed it as another kayer to Keras new Model

    this ofcourse require to match input dimension with original one



``` python
print("Building model with", MODULE_HANDLE)
model = tf.keras.Sequential([
    # Explicitly define the input shape so the model can be properly
    # loaded by the TFLiteConverter
    tf.keras.layers.InputLayer(input_shape=IMAGE_SIZE + (3,)),
    hub.KerasLayer(MODULE_HANDLE, trainable=do_fine_tuning),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(train_generator.num_classes,
                          kernel_regularizer=tf.keras.regularizers.l2(0.0001))
])
model.build((None,)+IMAGE_SIZE+(3,))
model.summary()
```


this shows how it is done.
https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/tf2_image_retraining.ipynb 



## using the Python Exported Trained model in Java script 

Tensor Flow allows us to convert model in library secific to platform which allows our backend to be seperated from the normal training process infact removing the need of backend altogether as all computation can now be shifted to user machine and only the end result are published on to the server side 

This is utility where all the converter are given 

[TensorFlow Converter All  ](https://github.com/tensorflow/tfjs )

[Availaible Converter List](https://github.com/tensorflow/tfjs-converter/tree/master/tfjs-converter)



## Availaible Model in Face-api.js 
[Availaible List](https://github.com/justadudewhohacks/face-api.js#available-models)
[Demo Here](https://justadudewhohacks.github.io/face-api.js/bbt_face_recognition/)

In the demo the only option is given in the face detection i.e. 
1. MTCNN
2. Tiny Face Detector 
3. SSD Mobile Net 

Rest for expression there is only one, face emebdding/ recogniton there is one -- Facenet 


## Table Example in MD file 
1 | 2 | 3 
--- | --- | ---
1 | 2 | 3 
1 | 2 | 3 
1 | 2 | 3 
