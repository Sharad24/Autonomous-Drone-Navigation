
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
DEFAULT_DTYPE = tf.float32


# In[2]:


def block(inputs,filters=64,strides=1):
    """
    Block 1 of Resnet 18
    
    Args:
    inputs: A tensor of size [batch, height_in, width_in, channels]
    filters: The number of filters for the convolutions.
    strides: The block's stride.
    """
    shortcut = inputs
    
    inputs = tf.layers.conv2d(inputs, filters, kernel_size=[3,3], strides=[1,strides,strides,1], 
                              padding='SAME',use_bias=False, 
                              kernel_initializer= tf.variance_scaling_initializer(factor=1.0, mode='FAN_IN'))
    
    inputs = tf.nn.selu(inputs)
    
    inputs = tf.layers.conv2d(inputs, filters, kernel_size=[3,3], strides=[1,strides,strides,1], 
                              padding='SAME',use_bias=False, 
                              kernel_initializer= tf.variance_scaling_initializer(factor=1.0, mode='FAN_IN'))
    
    
    
    inputs += shortcut
    
    inputs = tf.nn.selu(inputs)
    
    return inputs


# In[3]:


def block_down_sample(inputs,filters,strides=1):
    """
    Block 2 of Resnet 18
    
    Args:
    inputs: A tensor of size [batch, height_in, width_in, channels]
    filters: The number of filters for the convolutions.
    strides: The block's stride.
    """
    shortcut = inputs
    
    inputs = tf.layers.conv2d(inputs, filters, kernel_size=[3,3], strides=[1,strides,strides,1], 
                              padding='SAME',use_bias=False, 
                              kernel_initializer= tf.variance_scaling_initializer(factor=1.0, mode='FAN_IN'))
    
    inputs = tf.nn.selu(inputs)
    
    inputs = tf.layers.conv2d(inputs, filters, kernel_size=[3,3], strides=[1,2,2,1], 
                              padding='SAME',use_bias=False, 
                              kernel_initializer= tf.variance_scaling_initializer(factor=1.0, mode='FAN_IN'))
    
    inputs += shortcut
    
    inputs = tf.nn.selu(inputs)
    
    return inputs


# In[4]:



class Model(object):
  """Base class for building the Resnet Model."""

  def __init__(self, num_classes,
               dtype,data_format=None):
    """Creates a model for classifying an image.
    Args:
      num_classes: The number of classes used as labels.
      data_format: Input format ('channels_last', 'channels_first', or None).
        If set to None, the format is dependent on whether a GPU is available.
      dtype: The TensorFlow dtype to use for calculations. If not specified
        tf.float32 is used.
    Raises:
      ValueError: if invalid version is selected.
    """

    if not data_format:
      data_format = (
          'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

    if dtype not in ALLOWED_TYPES:
      raise ValueError('dtype must be one of: {}'.format(ALLOWED_TYPES))

    self.num_classes = num_classes
    self.data_format = data_format
    self.dtype = dtype

  def _custom_dtype_getter(self, getter, name, shape=None, dtype=DEFAULT_DTYPE,
                           *args, **kwargs):

    if dtype in CASTABLE_TYPES:
      var = getter(name, shape, tf.float32, *args, **kwargs)
      return tf.cast(var, dtype=dtype, name=name + '_cast')
    else:
      return getter(name, shape, dtype, *args, **kwargs)


  def _model_variable_scope(self):
    """Returns a variable scope that the model should be created under.
    If self.dtype is a castable type, model variable will be created in fp32
    then cast to self.dtype before being used.
    Returns:
      A variable scope for the model.
    """

    return tf.variable_scope('resnet_model',
                             custom_getter=self._custom_dtype_getter)

  def __call__(self, inputs, training):
    """Add operations to classify a batch of input images.
    Args:
      inputs: A Tensor representing a batch of input images.
      training: A boolean. Set to True to add operations required only when
        training the classifier.
    Returns:
      A logits Tensor with shape [<batch_size>, self.num_classes].
    """

    with self._model_variable_scope():
      if self.data_format == 'channels_last':
        # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
        # This provides a large performance boost on GPU. See
        # https://www.tensorflow.org/performance/performance_guide#data_formats
        inputs = tf.transpose(inputs, [0, 3, 1, 2])

      inputs = tf.layers.conv2d(inputs, 64, kernel_size=[7,7], strides=[1,2,2,1], 
                              padding='SAME',use_bias=False, 
                              kernel_initializer= tf.variance_scaling_initializer(factor=1.0, mode='FAN_IN'))
      inputs = tf.identity(inputs, 'initial_conv')

      inputs = tf.nn.selu(inputs)

      inputs = tf.layers.max_pooling2d( inputs=inputs, pool_size=[3,3], strides=[1,2,2,1], padding='SAME')
      inputs = tf.identity(inputs, 'initial_max_pool')
        
      inputs = block(inputs)
      inputs = tf.identity(inputs,'conv1a')
      inputs = block(inputs)
      inputs = tf.identity(inputs,'conv1b')
      
      for i in range(3):
        inputs = block_down_sample(inputs, 2**(i+7))
        inputs = block(inputs,2**(i+7))
        inputs = tf.identity(inputs,'conv{}'.format(i+2))

      # The current top layer has shape
      # `batch_size x pool_size x pool_size x final_size`.
      # ResNet does an Average Pooling layer over pool_size,
      # but that is the same as doing a reduce_mean. We do a reduce_mean
      # here because it performs better than AveragePooling2D.
      axes = [2, 3] if self.data_format == 'channels_first' else [1, 2]
      inputs = tf.reduce_mean(inputs, axes, keepdims=True)
      inputs = tf.identity(inputs, 'final_reduce_mean')

      inputs = tf.squeeze(inputs, axes)
      inputs = tf.layers.dense(inputs=inputs, units=self.num_classes)
      inputs = tf.identity(inputs, 'final_dense')
    return inputs

