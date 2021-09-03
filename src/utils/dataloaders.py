import os
import tensorflow as tf

__all__ = ['get_deepsea_dataset',]

def _get_tfr_data(datadir, compression_type='GZIP'):
    """
    datadir <str> - The directory with tfrecords shards.
    """
    assert os.path.exists(datadir)
    shards = [os.path.join(datadir, f) for f in os.listdir(datadir) if f.endswith(".tfrecords")]
    dataset = tf.data.TFRecordDataset(shards, compression_type=compression_type)
    return dataset

def _deepsea_pretrained_logits_decoder(example):
    schema = {
        "num_labels":tf.io.FixedLenFeature([], tf.int64),
        'y':tf.io.FixedLenFeature([], tf.string),
    }
    content = tf.io.parse_single_example(example, schema)
    num_labels = content['num_labels']
    y = tf.io.parse_tensor(content['y'], out_type=tf.float32)
    y = tf.reshape(y, shape=(num_labels,))
    return  y

def _deepsea_decoder(example):
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

def get_deepsea_dataset(datadir, compression_type='GZIP'):
    """
    datadir <str> -> The directory containing the train, test, validation splits
    of the deepsea data.

    The data should be in the form of tfrecords shards contained in 'train', 'test'
    and 'valid' folders for each split.


    RETURNS:

    dataset <dict> A dictionary containing the train test and validation splits as
    TFRecordsDataset objects.
    """
    splits = ['train', 'test', 'valid']
    dataset = {}
    for split in splits:
        dataset[split] = _get_tfr_data(
                                datadir=os.path.join(datadir, split),
                                compression_type=compression_type
                                    ).map(_deepsea_decoder)
    return dataset
