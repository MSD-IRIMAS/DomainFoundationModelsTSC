import tensorflow as tf
import math
import numpy as np

class PreTextDataLoader(tf.keras.utils.Sequence):

    def __init__(self, xtrain,
                       ytrain,
                       ytrain_ohe,
                       batch_size=64,
                       shuffle=True,):

        self.xtrain = xtrain
        self.ytrain = ytrain
        self.ytrain_ohe = ytrain_ohe

        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def __len__(self):
        return math.ceil(len(self.ytrain) / self.batch_size)
    
    def __getitem__(self, idx):

        low = idx * self.batch_size
        high = min(low + self.batch_size, len(self.ytrain))

        batch_x = self.xtrain[low:high]
        batch_y = self.ytrain[low:high]
        batch_y_ohe = self.ytrain_ohe[low:high]

        return [tf.keras.utils.pad_sequences(batch_x, dtype=np.float64), batch_y], batch_y_ohe
    
    def on_epoch_end(self):

        if self.shuffle:
            p = np.random.permutation(len(self.ytrain))
            self.xtrain = self.xtrain[p]
            self.ytrain = self.ytrain[p]
            self.ytrain_ohe = self.ytrain_ohe[p]