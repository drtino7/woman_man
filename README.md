<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
</head>
<body>
  <h1>Image Classification Model Training</h1>
  <p>This code is for training an image classification model. The model is trained on a dataset of images of men and women. The dataset is split into training and validation sets. The model is built using the MobileNetV2 architecture and trained using the Adam optimizer. The model is trained for 20 epochs.</p>

  <h2>Dataset</h2>
  <p>The dataset consists of images of men and women. The images are split into two directories: <code>mans</code> and <code>womans</code>. The images are preprocessed using the <code>ImageDataGenerator</code> class from the <code>tensorflow.keras.preprocessing.image</code> module. The images are rescaled, rotated, shifted, sheared, zoomed, and validated. The training and validation sets are split using the <code>validation_split</code> parameter.</p>

  <h2>Model</h2>
  <p>The model is built using the MobileNetV2 architecture. The MobileNetV2 architecture is a convolutional neural network that is designed for mobile devices. The architecture is optimized for low latency and low power consumption. The MobileNetV2 architecture is pre-trained on the ImageNet dataset. The pre-trained weights are used as the initial weights for the model. The last layer of the MobileNetV2 architecture is replaced with a dense layer with two output units. The dense layer is trained using the Adam optimizer. The loss function used is categorical cross-entropy. The model is trained for 20 epochs.</p>

  <h2>Results</h2>
  <p>The model achieves an accuracy of <strong> 0.9253%</strong> on the validation set after 20 epochs.</p>

  <pre><code>
import tensorflow as tf
import tensorflow_hub as tfh

url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
mobilenetv2 = tfh.KerasLayer(url,input_shape=(224,224,3))

#freezing layers
mobilenetv2.trainable = False

model = tf.keras.Sequential([
    mobilenetv2,
    tf.keras.layers.Dense(2,activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

epochs = 20
train = model.fit(
    datagen_train, epochs=epochs, batch_size=32,validation_data=datagen_test

)
  </code></pre>

  <p>Please let me know if you have any questions or need further assistance.</p>
</body>
</html>
