import tensorflow as tf
print(tf.__version__)
print(tf.test.is_built_with_cuda())  # If you have a GPU, it shows if it's built with CUDA
