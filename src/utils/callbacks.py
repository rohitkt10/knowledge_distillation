import numpy as np, os, pandas as pd
import tensorflow as tf
from tensorflow import keras as tfk

__all__ = ['ModelEvaluationCallback']

class ModelEvaluationCallback(tfk.callbacks.Callback):
    def __init__(self, x, y=None, batch_size=None, steps=None, filepath=None):
        super().__init__()
        self.filepath = filepath
        if not y:
            assert isinstance(x,tf.data.Dataset), "Either pass tf.data.Dataset object or both x, y tensors."
        if isinstance(x, tf.data.Dataset):
            batch_size = None
        else:
            if not batch_size:
                batch_size = 32
        self.options = {'x':x, 'y':y, 'batch_size':batch_size, 'return_dict':True, 'steps':steps}

    def on_epoch_end(self, epoch, logs=None):
        model = self.model
        res = model.evaluate(**self.options)
        printstr = "EVALUATION RESULTS: \n"
        for k in res.keys():
            printstr += f" - {k}: {res[k]}"
        print(printstr)

        # write results to disk
        if self.filepath:
            if not os.path.exists(os.path.dirname(self.filepath)):
                os.makedirs(os.path.dirname(self.filepath))
            df = pd.DataFrame(columns=res.keys(), data=np.array([list(res.values())]), index=[epoch+1])
            if not os.path.exists(self.filepath):
                print(f"Saving to location : {self.filepath}")
                df.to_csv(self.filepath,)
            else:
                _df = pd.read_csv(self.filepath, index_col=0)
                df = pd.concat([_df, df])
                df.to_csv(self.filepath,)
