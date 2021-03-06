from keras.backend.tensorflow_backend import set_session
from keras.backend.tensorflow_backend import clear_session
from keras.backend.tensorflow_backend import get_session
import tensorflow
import gc

def reset_keras():
    '''
    Perform garbage collection and manually release memory used by Keras.
    '''
    sess = get_session()
    if sess is not None:
        clear_session()
        sess.close()

    print(gc.collect())  # print out a number that indicates garbage collection

    config = tensorflow.ConfigProto()  # use the same config as you used to create the session
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = "0"
    set_session(tensorflow.Session(config=config))