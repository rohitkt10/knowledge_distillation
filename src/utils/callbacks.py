import numpy as np, os, pandas as pd
import tensorflow as tf
from tensorflow import keras as tfk
from ..distillation_strategies import basic_distiller

__all__ = ['ModelEvaluationCallback', 'RouteConstrainedCallback']

class ModelEvaluationCallback(tfk.callbacks.Callback):
    def __init__(self, x, y=None, batch_size=None, steps=None, filepath=None,):
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
        #print(type(model))
        #print(model)
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

class RouteConstrainedCallback(tfk.callbacks.Callback):
    def __init__(self, ckpts, patience=5, *args, **kwargs):
        """
        PARAMETERS
        """
        super().__init__(*args, **kwargs)
        self.ckpts = ckpts
        self.patience = patience

    def _remove_sigmoid(self, model):
        return tfk.Model(inputs=model.inputs, outputs=model.layers[-2].output, name=model.name)

    def on_train_begin(self, logs=None):
        ckpt = self.ckpts[0]
        teacher = tfk.models.load_model(ckpt)
        #if teacher.layers[-1].activation.__name__ == 'sigmoid':
        #    teacher = self._remove_sigmoid(teacher)
        teacher.trainable = False
        self.model.teacher = teacher

    def on_epoch_begin(self, epoch, logs=None):
        if epoch%self.patience == 0:
            idx = int(epoch/self.patience)
            if idx >= len(self.ckpts):
                ckpt = self.ckpts[-1]
            else:
                ckpt = self.ckpts[idx]
            print(f"Loading checkpoint saved at : {ckpt}")
            try:
                teacher = tfk.models.load_model(ckpt)
            except:
                #except Exception:
                print("Initial loading failed.")
                try:
                    teacher = tfk.models.load_model(ckpt)
                except:
                    print("Second attempt at loading checkpoint failed.")
                    raise RuntimeError
            #if teacher.layers[-1].activation.__name__ == 'sigmoid':
            #    teacher = self._remove_sigmoid(teacher)
            teacher.trainable = False
            self.model.teacher = teacher
