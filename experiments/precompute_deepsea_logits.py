"""
Precompute the logits from the pretrained deepsea model and
save them in TFRecords format.
"""

import numpy as np, os, sys
import tensorflow as tf

sys.path.append("..")
from src.utils import dataloaders
from src.utils import load_pretrained_teachers
from src.utils.dataloaders import _deepsea_decoder

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))): # if value ist tensor
        value = value.numpy() # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_array(array):
    array = tf.io.serialize_tensor(array)
    return array

def _encode_example(y,):
    """
    Set up single tf.train.Example instance.
    """
    num_labels = y.shape[0]
    feature = {
            "num_labels":_int64_feature(num_labels),
            "y":_bytes_feature(serialize_array(y))
        }
    features = tf.train.Features(feature=feature)
    example = tf.train.Example(features=features)
    return example

def write_data_to_tfrecord(y,
                           fname="data",
                           compression=False,
                           compression_level=4):
    """Takes a dataset (or a shard of a dataset) and writes it
    to a tfrecords file with specified compression settings"""
    if not fname.lower().endswith(".tfrecords"):
        fname= fname+".tfrecords"

    # set up the tfrecords writer
    if compression:
        options = tf.io.TFRecordOptions(compression_type="GZIP", compression_level=compression_level)
    else:
        options = None
    writer = tf.io.TFRecordWriter(fname, options=options)

    # iterate over all samples in the dataset
    for i in range(len(y)):
        example = _encode_example(y[i],).SerializeToString()
        writer.write(example)
    writer.close()

def main():
    # load the model
    model = load_pretrained_teachers.deepsea("../pretrained_models/deepsea/", logits_only=True)

    # get model predictions and save to tfrecords files
    savedir = "../data/deepsea/pretrained_logits"
    splits = ["train", "test", "valid"]
    for split in splits:
        splitdir = os.path.join(savedir, split)
        if not os.path.exists(splitdir):
            os.makedirs(splitdir)
        files = [f for f in os.listdir(f"../data/deepsea/{split}/") if f.endswith(".tfrecords")]
        for i, file in enumerate(files):
            print(f"Processing file : {file}")
            # load a tfrecord shard and get pretrained logit predictions
            f = os.path.join(f"../data/deepsea/{split}/", file)
            datashard = tf.data.TFRecordDataset([f], "GZIP").map(_deepsea_decoder)
            xdatashard = datashard.map(lambda x, y : x)
            ypred = model.predict(xdatashard.batch(64), verbose=1, )

            # save the pretrained
            fname = os.path.abspath(os.path.join(splitdir, file))
            write_data_to_tfrecord(ypred, fname=fname, compression=True)

if __name__ == '__main__':
    main()
