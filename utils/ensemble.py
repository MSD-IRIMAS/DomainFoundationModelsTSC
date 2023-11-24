import tensorflow as tf

def get_logits(model_dir, x):

    model = tf.keras.models.load_model(model_dir + 'best_model.hdf5',
                                       compile=False)
    
    return model.predict(x)