
import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variables
from tensorflow.python.util.tf_export import tf_export
def sobel(image):
  """Returns a tensor holding Sobel edge maps.
  Arguments:
    image: Image tensor with shape [batch_size, h, w, d] and type float32 or
    float64.  The image(s) must be 2x2 or larger.
  Returns:
    Tensor holding edge maps for each channel. Returns a tensor with shape
    [batch_size, h, w, d, 2] where the last two dimensions hold [[dy[0], dx[0]],
    [dy[1], dx[1]], ..., [dy[d-1], dx[d-1]]] calculated using the Sobel filter.
  """
  # Define vertical and horizontal Sobel filters.
  static_image_shape = image.get_shape()
  image_shape = array_ops.shape(image)
  kernels = [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
             [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
             [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]],
             [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]]]
  num_kernels = len(kernels)
  kernels = np.transpose(np.asarray(kernels), (1, 2, 0))
  kernels = np.expand_dims(kernels, -2)
  kernels_tf = constant_op.constant(kernels, dtype=image.dtype)

  kernels_tf = array_ops.tile(kernels_tf, [1, 1, image_shape[-1], 1],
                              name='sobel_filters')

  # Use depth-wise convolution to calculate edge maps per channel.
  pad_sizes = [[0, 0], [1, 1], [1, 1], [0, 0]]
  padded = array_ops.pad(image, pad_sizes, mode='REFLECT')

  # Output tensor has shape [batch_size, h, w, d * num_kernels].
  strides = [1, 1, 1, 1]
  output = nn.depthwise_conv2d(padded, kernels_tf, strides, 'VALID')

  # Reshape to [batch_size, h, w, d, num_kernels].
  shape = array_ops.concat([image_shape, [num_kernels]], 0)
  output = array_ops.reshape(output, shape=shape)
  output.set_shape(static_image_shape.concatenate([num_kernels]))
  return output
def sharpen(img):
    kernel_weights = np.array([[0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]], dtype=np.float32)
    # kernel_weights = kernel_weights / np.sum(kernel_weights)
    kernel_weights = np.reshape(kernel_weights, (*kernel_weights.shape, 1, 1))
    kernel_weights = np.tile(kernel_weights, (1, 1, img.get_shape().as_list()[3], 1))
    return tf.nn.depthwise_conv2d(img, kernel_weights, strides=[1, 1, 1, 1], padding="SAME")

def Sobel(x):
    dims = tf.shape (x)#return tf.squeeze( tf.image.sobel_edges(x))
    return tf.reshape(sobel(x),[dims[0],dims[1],dims[2],4])

def Sobel_shape(input_shape):
    dims = [input_shape[0],input_shape[1] ,input_shape[2] ,4]
    output_shape = tuple(dims)
    return output_shape
