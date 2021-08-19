"""
Define decoders for tfrecords datasets.

Example usage: 

>> decoder = get_decoder_fn(...)
>> dataset = tf.data.TFRecordsDataset(...) ## load the data from tfrecord files
>> dataset = dataset.map(decoder) ## decode the examples in the dataset; dataset now ready for use. 

The decoder is unique to the dataset. When setting up a dataset, write its
specific decoder and put it into this script.  
"""

import tensorflow as tf

__all__ = ['deepsea_decoder']

def deepsea_decoder(example):
    schema = {
        'length':tf.io.FixedLenFeature([], tf.int64),
        'depth':tf.io.FixedLenFeature([], tf.int64),
        "num_labels":tf.io.FixedLenFeature([], tf.int64),
        'y':tf.io.FixedLenFeature([], tf.string),
        'x':tf.io.FixedLenFeature([], tf.string), 
    }
    content = tf.io.parse_single_example(example, schema)
    length, depth, num_labels = content['length'], content['depth'], content['num_labels']
    x = tf.io.parse_tensor(content['x'], out_type=tf.uint8)
    x = tf.reshape(x, shape=(length, depth))
    x = tf.cast(x, tf.float32)

    y = tf.io.parse_tensor(content['y'], out_type=tf.uint8)
    y = tf.reshape(y, shape=(num_labels,))
    y = tf.cast(y, dtype=x.dtype)
    return  x, y