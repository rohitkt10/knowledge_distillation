import tensorflow as tf
from tensorflow import keras as tfk

def get_model(model_name):
    assert model_name in available_models, "No such pretrained model available. \
                                            Please pick one of "+str(available_models)


def deepsea(ckptdir, *args, **kwargs):
    ckptdir = os.path.abspath(ckptdir)
    assert os.path.exists(os.path.join(ckptdir, "saved_model.pb")), \
                    "No saved model located in specified directory."
    model = tfk.models.load_model(
                                filepath=ckptdir,
                                *args,
                                **kwargs,
                                 )
    return model

def danq(ckptdir, compile=True, *args, **kwargs):
    ckptdir = os.path.abspath(ckptdir)
    assert os.path.exists(os.path.join(ckptdir, "saved_model.pb")), \
                    "No saved model located in specified directory."
    model = tfk.models.load_model(
                                filepath=ckptdir,
                                *args,
                                **kwargs,
                                 )
    return model
