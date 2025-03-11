import tensorflow as tf

# Check TensorFlow version
print(tf.__version__)

# Check if GPU is available
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

# Print the GPU device name
print(tf.test.gpu_device_name())