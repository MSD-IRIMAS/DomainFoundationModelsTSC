import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import OneHotEncoder as OHE
from sklearn.metrics import accuracy_score

from utils.utils import create_directory

class HINCEPTION_BASELINE:

    def __init__(self,
                 output_dir,
                 length_TS,
                 n_classes,
                 depth=3,
                 batch_size=64,
                 n_epochs=1000,):

        self.output_dir = output_dir

        self.length_TS = length_TS
        self.n_classes = n_classes

        self.batch_size = batch_size
        self.n_epochs = n_epochs

        self.depth = depth

        self.build_model()
    
    def _hybrid_layer(self,input_tensor,input_channels,kernel_sizes=[2,4,8,16,32,64]):
    

        '''
        Function to create the hybrid layer consisting of non trainable Conv1D layers with custom filters.
        Args:
            input_tensor: input tensor
            input_channels : number of input channels, 1 in case of UCR Archive
        '''

        conv_list = []

        # for increasing detection filters

        for kernel_size in kernel_sizes:

            filter_ = np.ones(shape=(kernel_size,input_channels,1)) # define the filter weights with the shape corresponding the Conv1D layer in keras (kernel_size, input_channels, output_channels)
            indices_ = np.arange(kernel_size)

            filter_[indices_ % 2 == 0] *= -1 # formula of increasing detection filter

            # Create a Conv1D layer with non trainable option and no biases and set the filter weights that were calculated in the line above as the initialization

            conv = tf.keras.layers.Conv1D(filters=1,kernel_size=kernel_size,padding='same',
                                          use_bias=False,kernel_initializer=tf.keras.initializers.Constant(filter_),
                                          trainable=False,name='hybrid-increasse-'+str(self.keep_track)+'-'+str(kernel_size))(input_tensor)

            conv_list.append(conv) # add the conv layer to the list

            self.keep_track += 1

        # for decreasing detection filters
        
        for kernel_size in kernel_sizes:

            filter_ = np.ones(shape=(kernel_size,input_channels,1)) # define the filter weights with the shape corresponding the Conv1D layer in keras (kernel_size, input_channels, output_channels)
            indices_ = np.arange(kernel_size)

            filter_[indices_ % 2 > 0] *= -1 # formula of decreasing detection filter

            # Create a Conv1D layer with non trainable option and no biases and set the filter weights that were calculated in the line above as the initialization

            conv = tf.keras.layers.Conv1D(filters=1,kernel_size=kernel_size,padding='same',
                                          use_bias=False,kernel_initializer=tf.keras.initializers.Constant(filter_),
                                          trainable=False,name='hybrid-decrease-'+str(self.keep_track)+'-'+str(kernel_size))(input_tensor)
            
            conv_list.append(conv) # add the conv layer to the list

            self.keep_track += 1

        # for peak detection filters
        
        for kernel_size in kernel_sizes[1:]:

            filter_ = np.zeros(shape=(kernel_size + kernel_size // 2,input_channels,1))

            xmesh = np.linspace(start=0,stop=1,num=kernel_size//4+1)[1:].reshape((-1,1,1))

            # see utils.custom_filters.py to understand the formulas below

            filter_left = xmesh**2
            filter_right = filter_left[::-1]

            filter_[0:kernel_size // 4] = -filter_left
            filter_[kernel_size // 4:kernel_size // 2] = -filter_right
            filter_[kernel_size // 2:3 * kernel_size // 4] = 2 * filter_left
            filter_[3 * kernel_size // 4:kernel_size] = 2 * filter_right
            filter_[kernel_size:5 * kernel_size // 4] = -filter_left
            filter_[5 * kernel_size // 4:] = -filter_right
            
            # Create a Conv1D layer with non trainable option and no biases and set the filter weights that were calculated in the line above as the initialization

            conv = tf.keras.layers.Conv1D(filters=1,kernel_size=kernel_size+kernel_size//2,padding='same',
                                          use_bias=False,kernel_initializer=tf.keras.initializers.Constant(filter_),
                                          trainable=False,name='hybrid-peeks-'+str(self.keep_track)+'-'+str(kernel_size))(input_tensor)

            conv_list.append(conv) # add the conv layer to the list

            self.keep_track += 1

        
        hybrid_layer = tf.keras.layers.Concatenate(axis=2, name='concatenate-hybrid')(conv_list) # concantenate all convolution layers
        hybrid_layer = tf.keras.layers.Activation(activation='relu', name='activation-hybrid')(hybrid_layer) # apply activation ReLU

        return hybrid_layer

    def _inception_module(self, input_tensor,
                                n_conv_layers=3,
                                n_filters=32,
                                kernel_size=40,
                                use_custom_filters=False,
                                pool_size=3):
        
        if int(input_tensor.shape[-1]) > 1:

            input_inception = tf.keras.layers.Conv1D(filters=n_filters,
                                                     kernel_size=1,
                                                     padding='same',
                                                     use_bias=False)(input_tensor)
        
        else:
            input_inception = input_tensor
        
        kernel_sizes = [kernel_size // (2 ** n) for n in range(n_conv_layers)]

        conv_layers = [tf.keras.layers.Conv1D(filters=n_filters,
                                              kernel_size=kernel_sizes[n],
                                              use_bias=False,
                                              padding='same')(input_inception) for n in range(n_conv_layers)]
        
        max_pool = tf.keras.layers.MaxPool1D(pool_size=pool_size,
                                             strides=1,
                                             padding='same')(input_tensor)

        max_pool = tf.keras.layers.Conv1D(filters=n_filters,
                                          kernel_size=1,
                                          padding='same',
                                          use_bias=False)(max_pool)
        
        conv_layers.append(max_pool)

        if use_custom_filters:

            hybrid_layer = self._hybrid_layer(input_tensor=input_tensor,
                                             input_channels=int(input_tensor.shape[-1]))
        
            conv_layers.append(hybrid_layer)
        
        x = tf.keras.layers.Concatenate(axis=-1)(conv_layers)

        batch_norm = tf.keras.layers.BatchNormalization()(x)
    
        activation = tf.keras.layers.Activation(activation='relu')(batch_norm)
        return activation
    
    def _shortcut_layer(self, input_tensor,
                              out_tensor):

        # Function to add residual connection between input and output tensors

        shortcut_y = tf.keras.layers.Conv1D(filters=int(out_tensor.shape[-1]),
                                            kernel_size=1,
                                            padding='same',
                                            use_bias=False)(input_tensor)

        batch_norm = tf.keras.layers.BatchNormalization()(shortcut_y)

        x = tf.keras.layers.Add()([batch_norm, out_tensor])
        x = tf.keras.layers.Activation('relu')(x)

        return x

    def _inception_block(self, input_tensor,
                               depth=3,
                               first_block=False):
        
        x = input_tensor
        input_res = input_tensor

        use_custom_filters = False

        if first_block:
            use_custom_filters = True
        
        for d in range(depth):

            if d == 0 and use_custom_filters:
                x = self._inception_module(input_tensor=x, use_custom_filters=use_custom_filters)

            else:
                x = self._inception_module(input_tensor=x)

        x = self._shortcut_layer(input_tensor=input_res,
                                 out_tensor=x)
                
        return x

    def build_model(self, compile_model=True, return_model=False):

        self.keep_track = 0

        input_layer = tf.keras.layers.Input((self.length_TS,))
        reshape_layer = tf.keras.layers.Reshape(target_shape=(self.length_TS, 1))(input_layer)

        x = reshape_layer

        x = self._inception_block(input_tensor=x,
                                  depth=self.depth,
                                  first_block=True)
        
        x = self._inception_block(input_tensor=x,
                                  depth=self.depth,
                                  first_block=False)
        
        gap = tf.keras.layers.GlobalAveragePooling1D()(x)

        output_layer = tf.keras.layers.Dense(units=self.n_classes,
                                             use_bias=False,
                                             activation='softmax')(gap)
        
        self.model = tf.keras.models.Model(inputs=input_layer,
                                           outputs=output_layer)
        
        if compile_model:

            self.model.compile(optimizer='Adam',
                               loss='categorical_crossentropy',
                               metrics=['accuracy'])
            
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50,
                                                                min_lr=1e-4)
        
            model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=self.output_dir+'best_model.h5',
                                                                  monitor='loss', save_best_only=True, save_weights_only=True)
            
            self.callbacks = [reduce_lr, model_checkpoint]
        
        if return_model:
            return self.model
    
    def fit(self, xtrain, ytrain):

        ohe = OHE(sparse=False)
        ytrain = np.expand_dims(ytrain, axis=1)
        ytrain = ohe.fit_transform(ytrain)

        self.model.fit(xtrain,
                       ytrain,
                       batch_size=self.batch_size,
                       epochs=self.n_epochs,
                       callbacks=self.callbacks)
        
        tf.keras.backend.clear_session()
    
    def predict(self, xtest, ytest):

        model = self.build_model(compile_model=False, return_model=True)
        model.load_weights(self.output_dir+'best_model.h5')

        logits = model.predict(xtest)
        ypred = np.argmax(logits, axis=1)

        df = pd.DataFrame(columns=['accuracy'])

        df['accuracy'] = [accuracy_score(y_true=ytest,
                                         y_pred=ypred,
                                         normalize=True)]
        
        df.to_csv(self.output_dir+'metrics.csv', index=False)

        tf.keras.backend.clear_session()

        return logits