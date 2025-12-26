from tensorflow.python.keras.models import load_model
import tensorflow as tf


model = load_model('models/main_model.h5')

# regular convert
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_regular_model = converter.convert()

with open("converted_models/regular_model.tflite", 'wb') as f:
  f.write(tflite_regular_model)


# float16 converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_fp16_model = converter.convert()

with open("converted_models/float16_model.tflite", 'wb') as f:
  f.write(tflite_fp16_model)