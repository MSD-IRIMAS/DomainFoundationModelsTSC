import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import OneHotEncoder as OHE
from sklearn.metrics import accuracy_score

from utils.utils import create_directory
from utils.data_loader import PreTextDataLoader

from tqdm.keras import TqdmCallback

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

class HINCEPTION:

    def __init__(self,
                 output_dir,
                 list_of_datasets,
                 list_of_n_classes,
                 list_of_length_TS,
                 n_inception_blocks_pretext=1,
                 n_inception_blocks=1,
                 depth_pretext=3,
                 depth=3,
                 batch_size_pretext=64,
                 n_epochs_pretext=1000,
                 batch_size=64,
                 n_epochs=1000,):

        self.output_dir = output_dir

        self.list_of_datasets = list_of_datasets
        self.n_datasets = len(self.list_of_datasets)

        self.batch_size_pretext = batch_size_pretext
        self.n_epochs_pretext = n_epochs_pretext

        self.batch_size = batch_size
        self.n_epochs = n_epochs

        self.list_of_n_classes = list_of_n_classes
        self.list_of_length_TS = list_of_length_TS

        self.max_length = np.max(self.list_of_length_TS)

        self.n_inception_blocks_pretext = n_inception_blocks_pretext
        self.depth_pretext = depth_pretext

        self.n_inception_blocks = n_inception_blocks
        self.depth = depth

        self.keep_track = 0

        self.build_model_pretext()
    
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
    
    def _inception_module_pretext(self, input_tensor,
                                  input_layer_dataset,
                                  count_bn,
                                  n_conv_layers=3,
                                  n_filters=32,
                                  kernel_size=40,
                                  use_custom_filters=False,
                                  pool_size=3):
        
        if int(input_tensor.shape[-1]) > 1:

            input_inception = tf.keras.layers.Conv1D(filters=n_filters,
                                                     kernel_size=1,
                                                     padding='same',
                                                     use_bias=False,
                                                     name=tf.compat.v1.get_default_graph().unique_name('bottleneck'))(input_tensor)
        
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
                                          use_bias=False,
                                          name=tf.compat.v1.get_default_graph().unique_name("conv-max-pooling1d"))(max_pool)
        
        conv_layers.append(max_pool)

        if use_custom_filters:

            hybrid_layer = self._hybrid_layer(input_tensor=input_tensor,
                                             input_channels=int(input_tensor.shape[-1]))
            conv_layers.append(hybrid_layer)
        
        x = tf.keras.layers.Concatenate(axis=-1)(conv_layers)

        batch_norms = tf.stack([tf.keras.layers.BatchNormalization(name='batch-norm-'+str(
            count_bn)+'-'+self.list_of_datasets[d])(x) for d in range(self.n_datasets)])
        
        batch_norms = tf.transpose(batch_norms, [1,2,3,0])
        
        chosen_batch_norm = tf.gather(batch_norms, input_layer_dataset, axis=-1, batch_dims=0)
        chosen_batch_norm = chosen_batch_norm[:,:,:,0,0]
        
        activation = tf.keras.layers.Activation(activation='relu')(chosen_batch_norm)
        return activation
    
    def _shortcut_layer_pretext(self, input_tensor,
                                out_tensor,
                                input_layer_dataset):

        # Function to add residual connection between input and output tensors

        shortcut_y = tf.keras.layers.Conv1D(filters=int(out_tensor.shape[-1]),
                                            kernel_size=1,
                                            padding='same',
                                            use_bias=False,
                                            name='shortcut-conv')(input_tensor)

        batch_norms = tf.stack([tf.keras.layers.BatchNormalization(name='batch-norm-residual-'+
            '-'+self.list_of_datasets[d])(shortcut_y) for d in range(self.n_datasets)])
        
        batch_norms = tf.transpose(batch_norms, [1,2,3,0])
        
        chosen_batch_norm = tf.gather(batch_norms, input_layer_dataset, axis=-1, batch_dims=0)
        chosen_batch_norm = chosen_batch_norm[:,:,:,0,0]

        x = tf.keras.layers.Add()([chosen_batch_norm, out_tensor])
        x = tf.keras.layers.Activation('relu', name='shortcut-activation')(x)

        return x
    
    def _inception_block_pretext(self, input_tensor,
                                 input_layer_dataset,
                                 count_bn,
                                 depth=3):
        
        x = input_tensor
        input_res = input_tensor
        
        for d in range(depth):

            if d == 0 and count_bn == 0:
                x = self._inception_module_pretext(input_tensor=x,
                                           input_layer_dataset=input_layer_dataset,
                                           count_bn=count_bn,
                                           use_custom_filters=True)
            
            else:
                x = self._inception_module_pretext(input_tensor=x,
                                           input_layer_dataset=input_layer_dataset,
                                           count_bn=count_bn,
                                           use_custom_filters=False)
            
            count_bn += 1

        x = self._shortcut_layer_pretext(input_tensor=input_res,
                                    out_tensor=x,
                                    input_layer_dataset=input_layer_dataset)
                
        return x, count_bn
    
    def build_model_pretext(self, compile_model=True,
                                  return_model=False):

        input_layer = tf.keras.layers.Input((None,))
        input_layer_dataset = tf.keras.layers.Input((1,), dtype=tf.int32)

        reshape_layer = tf.keras.layers.Reshape(
            target_shape=(-1, 1))(input_layer)

        x = reshape_layer

        count_bn = 0


        x, count_bn_ = self._inception_block_pretext(input_tensor=x,
                                    input_layer_dataset=input_layer_dataset,
                                    count_bn=count_bn,
                                    depth=self.depth_pretext)
            
        count_bn += count_bn_
        
        gap = tf.keras.layers.GlobalAveragePooling1D()(x)

        output_layer = tf.keras.layers.Dense(units=self.n_datasets,
                                             use_bias=False,
                                             activation='softmax')(gap)
        
        self.model_pretext = tf.keras.models.Model(inputs=[input_layer, input_layer_dataset],
                                                   outputs=output_layer)
        
        if compile_model:

            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50,
                                                             min_lr=1e-4)

            model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=self.output_dir+'best_model_pretext.h5',
                                                                  monitor='loss', save_best_only=True, save_weights_only=True)

            self.callbacks_pretext = [reduce_lr, model_checkpoint, TqdmCallback(verbose=1)]

            self.model_pretext.compile(loss='categorical_crossentropy',
                               optimizer='Adam',
                               metrics=['accuracy'])
            
            # tf.keras.utils.plot_model(self.model_pretext, self.output_dir+'model.pdf')

        if return_model:
            return self.model_pretext
    
    def fit_pretext(self, xtrains, ytrains):

        ohe = OHE(sparse=False)
        ytrain_ohe = np.expand_dims(ytrains, axis=1)
        ytrain_ohe = ohe.fit_transform(ytrain_ohe)

        seq_loader = PreTextDataLoader(xtrain=np.array(xtrains, dtype=object),
                                       ytrain=np.expand_dims(ytrains, axis=1), 
                                       ytrain_ohe=ytrain_ohe,
                                       batch_size=self.batch_size_pretext)

        self.model_pretext.fit(seq_loader,
                               batch_size=self.batch_size_pretext,
                               epochs=self.n_epochs_pretext,
                               callbacks=self.callbacks_pretext,
                               verbose=0)
    
        tf.keras.backend.clear_session()
    
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
                               depth=3):
        
        x = input_tensor
        input_res = input_tensor
        
        for d in range(depth):

            x = self._inception_module(input_tensor=x)

        x = self._shortcut_layer(input_tensor=input_res,
                                 out_tensor=x)
                
        return x

    def _build_models(self, _output_dir):

        self.models = []
        self.callbacks = []

        pretext_model = self.build_model_pretext(compile_model=False,
                                                 return_model=True)

        pretext_model.load_weights(self.output_dir+'best_model_pretext.h5')

        self.callbacks = []

        for d in range(self.n_datasets):

            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50,
                                                                min_lr=1e-4)
            
            create_directory(_output_dir+self.list_of_datasets[d]+'/')

            model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=_output_dir+self.list_of_datasets[d]+'/best_model.hdf5',
                                                                  monitor='loss',
                                                                  save_best_only=True)

            self.callbacks.append([reduce_lr, model_checkpoint, TqdmCallback(verbose=1)])

            input_layer = tf.keras.layers.Input((self.list_of_length_TS[d],))

            x = input_layer

            layers = pretext_model.layers
            pretrained_layers = []

            for l in range(len(layers)):

                if 'global_average_pooling1d' in layers[l].name:
                    continue
                
                if 'dense' in layers[l].name:
                    continue
            
                if 'input' in layers[l].name:
                    continue
            
                if 'stack' in layers[l].name:
                    continue
            
                if 'transpose' in layers[l].name:
                    continue

                if 'gather' in layers[l].name:
                    continue
            
                if 'getitem' in layers[l].name:
                    continue

                if 'batch-norm' in layers[l].name and self.list_of_datasets[d] in layers[l].name:
                    pretrained_layers.append(layers[l])
                
                elif 'batch-norm' in layers[l].name:
                    continue
            
                else:
                    pretrained_layers.append(layers[l])

            hybrid_layers = []
            conv_layers_inception = []
            hybrid_concat = 0
            hybrid_activation = 0
            max_pool = 0
            input_max_pool = 0
            shortcut = 0
            
            for layer in pretrained_layers:
                print(layer.name)
            # exit()

            for layer in pretrained_layers:

                if layer.name == 'concatenate-hybrid':
                    hybrid_concat = layer(hybrid_layers)

                elif layer.name == 'activation-hybrid':
                    hybrid_activation = layer(hybrid_concat)

                # elif layer.name == 'concat-hybrid-inception':
                #     x = layer([x, hybrid_activation])

                elif 'hybrid' in layer.name:
                    hybrid_layers.append(layer(reshape))
                
                elif 'max_pooling1d' in layer.name:
                    max_pool = layer(input_max_pool)

                elif 'conv-max-pooling1d' in layer.name:
                    conv_layers_inception.append(layer(max_pool))

                elif layer.name == 'shortcut-conv':
                    shortcut = layer(reshape)
                
                elif 'batch-norm-residual' in layer.name:
                    shortcut = layer(shortcut)
                
                elif 'add' in layer.name:
                    x = layer([x, shortcut])
                
                elif layer.name == 'shortcut-activation':
                    x = layer(x)

                elif 'conv1d' in layer.name:
                    conv_layers_inception.append(layer(x))
                
                elif 'concatenate' in layer.name:
                    if hybrid_activation is not None:
                        conv_layers_inception.append(hybrid_activation)
                        hybrid_activation = None
                    x = layer(conv_layers_inception)
                    conv_layers_inception = []
                
                elif 'batch-norm' in layer.name:
                    x = layer(x)
                
                elif 'activation' in layer.name:
                    x = layer(x)
                
                elif 'reshape' in layer.name:
                    x = layer(x)
                    reshape = x
                    input_max_pool = x
                
                elif 'bottleneck' in layer.name:
                    input_max_pool = x
                    x = layer(x)
                
            x = self._inception_block(input_tensor=x, depth=self.depth)

            x = tf.keras.layers.GlobalAveragePooling1D()(x)

            output_layer = tf.keras.layers.Dense(units=self.list_of_n_classes[d],
                                                 use_bias=False,
                                                 activation='softmax')(x)
            
            self.models.append(tf.keras.models.Model(inputs=input_layer,
                                                     outputs=output_layer))
            
            self.models[-1].compile(loss='categorical_crossentropy',
                                    optimizer='Adam',
                                    metrics=['accuracy'])

            # tf.keras.utils.plot_model(self.models[-1], _output_dir+self.list_of_datasets[d]+'/model.pdf')

    def _fit_models(self, xtrains, ytrains):

        for d in range(self.n_datasets):

            xtrain = xtrains[d]
            ytrain = ytrains[d]

            ohe = OHE(sparse=False)
            ytrain = np.expand_dims(ytrain, axis=1)
            ytrain = ohe.fit_transform(ytrain)

            self.models[d].fit(xtrain,
                               ytrain,
                               batch_size=self.batch_size,
                               epochs=self.n_epochs,
                               callbacks=self.callbacks[d],
                               verbose=0)
    
    def _predict_models(self, xtests, ytests, _output_dir, ypreds):

        for d in range(self.n_datasets):

            print(self.list_of_datasets[d])

            df = pd.DataFrame(columns=['accuracy'])

            model = tf.keras.models.load_model(_output_dir+self.list_of_datasets[d]+'/best_model.hdf5', compile=False)

            logits = model.predict(xtests[d])
            ypred = np.argmax(logits, axis=1)

            score = accuracy_score(y_true=ytests[d],
                                   y_pred=ypred,
                                   normalize=True)
            
            df['accuracy'] = [score]

            df.to_csv(_output_dir+self.list_of_datasets[d]+'/metrics.csv', index=False)

            ypreds[d] = ypreds[d] + [logits]
    
        return ypreds

    def fit_and_predict_models(self, xtrains, ytrains, xtests, ytests, n_runs=5, train_models=True):

        ypreds = [[] for _ in range(self.n_datasets)]

        for _run_fine_tune in range(n_runs):

            _output_dir_run = self.output_dir + 'fine_tune_run_' + str(_run_fine_tune) + '/'
            create_directory(_output_dir_run)

            if train_models:

                self._build_models(_output_dir=_output_dir_run)

                self._fit_models(xtrains=xtrains,
                                ytrains=ytrains)
            
            ypreds = self._predict_models(xtests=xtests,
                                          ytests=ytests,
                                          _output_dir=_output_dir_run,
                                          ypreds=ypreds)
        
        ypreds_np = [0 for _ in range(self.n_datasets)]

        for d in range(self.n_datasets):

            ypreds_np[d] = np.zeros(shape=(len(ytests[d]), self.list_of_n_classes[d]))
            
            for ypred in ypreds[d]:

                ypreds_np[d] += np.asarray(ypred)
        
        del ypreds

        return ypreds_np

if __name__ == '__main__':

    inc = HINCEPTION(output_dir='/run/user/1001/gvfs/sftp:host=10.59.251.111,port=2222,user=afawaz/media/afawaz/DATA/ea666698-a6cb-40ab-9a39-645c92f0c70a/home/hadi/phd/things_to_try/Pretext_Task_Under_Construction/',
    list_of_datasets=['data1','data2'],
    list_of_length_TS=[100,120],
    list_of_n_classes=[2,2])

    inc.model_pretext.summary()