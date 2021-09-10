import os
import tensorflow as tf
from tensorflow import keras as tfk

__all__ = ['deepsea', 'danq']

def deepsea(ckptdir, logits_only=False, *args, **kwargs):
    ckptdir = os.path.abspath(ckptdir)
    assert os.path.exists(os.path.join(ckptdir, "saved_model.pb")), \
                    "No saved model located in specified directory."
    model = tfk.models.load_model(
                                filepath=ckptdir,
                                *args,
                                **kwargs,
                                 )
    if logits_only:
        model.pop() ## remove the sigmoid activation last layer
    return model

def danq(ckptdir, logits_only=False, *args, **kwargs):
    ckptdir = os.path.abspath(ckptdir)
    assert os.path.exists(os.path.join(ckptdir, "saved_model.pb")), \
                    "No saved model located in specified directory."
    model = tfk.models.load_model(
                                filepath=ckptdir,
                                *args,
                                **kwargs,
                                 )
    if logits_only:
        model.pop() ## remove the sigmoid activation last layer
    return model
