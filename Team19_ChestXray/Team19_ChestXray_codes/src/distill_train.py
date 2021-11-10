import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import vgg16, vgg19, mobilenet_v2, inception_v3, nasnet, inception_resnet_v2
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, AvgPool2D, \
                         Lambda, Dropout, GlobalAveragePooling2D, multiply, LocallyConnected2D, BatchNormalization
from keras.models import Sequential, Model
from keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau

import params
import preprocessing_pyspark
import reset
import gradient_accumulation
from utils import plot_train_metrics, plot_ROC, save_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class CNN_Model():
    def __init__(self, transfer_model, image_size, preprocess_input, custom_model_name, fc_output, optimizer):
      self.transferModel = transfer_model
      self.image_size = image_size
      self.preprocess_input = preprocess_input
      self.custom_model_name = custom_model_name
      self.fc_output = fc_output
      self.optimizer = optimizer
      self.base_model = self.create_base_model()
      self.custom_model = self.create_custom_model()
      self.final_model = self.create_final_model()
      self.model_name = f'{self.custom_model_name}_{self.base_model.name}'
     
    def create_base_model(self):
        '''
            Create a keras application base model for transfer learning
        '''
        base_model = self.transferModel(weights='imagenet', 
                                        include_top=False, 
                                        input_shape=(self.image_size[0], self.image_size[1], 3)) # rgb mode, dim = 3
        base_model.trainable = False
        return base_model

    def create_custom_model(self):
        ''' 
        Create custom model
        '''
        if self.custom_model_name == 'simple':
            custom_model = self._create_simple_model(self.fc_output)
        elif self.custom_model_name == 'attention':
            custom_model = self._create_attention_model(self.base_model, self.fc_output)
        else:
            raise NameError("""Not valid custom layer, choose 'simple' or 'attention'""")
        return custom_model

    def create_final_model(self):
        '''
            Creates final model for the task
        '''
        model = Sequential()
        model.add(self.base_model)
        model.add(self.custom_model)
        model.compile(optimizer=self.optimizer,
                      loss='binary_crossentropy',
                      metrics=['binary_accuracy', 'mae'])
        print(f'{model.summary()}')
        return model

    @staticmethod
    def _create_simple_model(fc_output):
        '''
        Creates a simple model by adding dropout, pooling, and dense layer to a pretrained model
      
        Args:
          base_model: keras base model
          disease_classes: 
          optimizer: 
      
        Returns:
          The created Model.
      
        '''
        model = Sequential()
        model.add(GlobalAveragePooling2D())
        model.add(Dropout(0.5))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(fc_output, activation='sigmoid'))
        model.name='simple_model'
        return model

    @staticmethod
    def _create_attention_model(base_model, fc_output):
        '''
        Creates an attention model by adding attention layers to base_model
      
        Args:
          base_model: The keras Base model at the start
          disease_classes: The labels to use
          optimizer: The optimizer to use
      
        Returns:
           attention model.
      
        '''
        frozen_features = Input(base_model.get_output_shape_at(0)[1:], name='feature_input')
        frozen_depth = base_model.get_output_shape_at(0)[-1]
        new_features = BatchNormalization()(frozen_features)
      
        # here we do an attention mechanism to turn pixels in the GAP on an off
        attention_layer_1 = Conv2D(64, kernel_size=(1, 1), padding='same', activation='elu')(new_features)
        attention_layer_2 = Conv2D(32, kernel_size=(1, 1), padding='same', activation='elu')(attention_layer_1)
        attention_layer_3 = AvgPool2D((2,2), strides=(1,1), padding='same')(attention_layer_2)
        attention_layer_4 = Conv2D(16, kernel_size=(1, 1), padding='same', activation='elu')(attention_layer_3)
        attention_layer_5 = AvgPool2D((2, 2), strides=(1, 1), padding='same')(attention_layer_4)  # smooth results
        attention_layer_6 = Conv2D(1, kernel_size=(1, 1), padding='valid', activation='sigmoid')(attention_layer_5)
      
        # fan it out to all of the channels
        up_c2_w = np.ones((1, 1, 1, frozen_depth))
        up_c2 = Conv2D(frozen_depth, kernel_size=(1, 1), padding='same', activation='linear', use_bias=False, weights=[up_c2_w])
        up_c2.trainable = False
        attention_layer_7 = up_c2(attention_layer_6)
      
        mask_features = multiply([attention_layer_7, new_features])
        gap_features = GlobalAveragePooling2D()(mask_features)
        gap_mask = GlobalAveragePooling2D()(attention_layer_6)
      
        # to account for missing values from the attention model
        gap = Lambda(lambda x: x[0]/x[1], name='RescaleGAP')([gap_features, gap_mask])
        gap_dr = Dropout(0.5)(gap)
        dr_steps = Dropout(0.5)(Dense(128, activation='elu')(gap_dr))
        out_layer = Dense(fc_output, activation='sigmoid')(dr_steps)
      
        # creating the attention model
        attention_model = Model(inputs=[frozen_features], 
                                outputs=[out_layer], 
                                name='attention_model')
        return attention_model


def create_data_generator(metadata,
                          disease_classes,
                          batch_size,
                          target_size=params.IMG_SIZE):
    '''
    Create a keras DataGenerator for the input dataset
    Args:
      dataset: The images subset to use
      disease_classes: The labels to use
      batch_size: The batch_size of the generator
      target_size: The (x, y) image size to scale the images
    Return:
      The created ImageDataGenerator.
    '''

    image_generator = ImageDataGenerator(samplewise_center=True,
                                         samplewise_std_normalization=True,
                                         horizontal_flip=True,
                                         vertical_flip=False,
                                         height_shift_range=0.05,
                                         width_shift_range=0.1,
                                         #brightness_range=[0.7, 1.5],
                                         rotation_range=5,
                                         shear_range=0.1,
                                         fill_mode='reflect',
                                         zoom_range=0.15)

    # metadata associated with image
    data_generator = image_generator.flow_from_dataframe(dataframe=metadata,
                                                         directory=None,
                                                         x_col='path',
                                                         y_col='labels',
                                                         class_mode='categorical',
                                                         classes=disease_classes,
                                                         target_size=target_size,
                                                         color_mode='rgb',
                                                         batch_size=batch_size)

    return data_generator


def configure_callbacks(results_folder, model_name):

    weight_path = os.path.join(results_folder, params.WEIGHT_FILE_NAME)
  
    checkpoint = ModelCheckpoint(weight_path,
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='min',
                                 save_weights_only=True)

    early = EarlyStopping(monitor="val_loss",
                          mode="min",
                          patience=params.EARLY_STOPPING_PATIENCE)

    tensorboard = TensorBoard(log_dir=os.path.join(params.RESULTS_FOLDER, params.TENSORBOARD_BASE_FOLDER, model_name))
    dynamicLR = ReduceLROnPlateau(monitor='val_loss', 
                                  factor=0.2,
                                  patience=params.RL_PLATEAU_PATIENCE, 
                                  min_lr=params.LEARNING_RATE/100)
    callbacks_list = [tensorboard, checkpoint, dynamicLR, early]
    return callbacks_list


def run_procedure():
    '''
    main procedure logics
    '''
    # metadata loading
    metadata, disease_classes = preprocessing_pyspark.preprocess_metadata()
    train, valid, test = preprocessing_pyspark.stratify_train_test_split(metadata)

    """
    list of base models
    ref: https://keras.io/api/applications/
    """
    base_models = [
        (vgg16.VGG16, params.VGG16_IMG_SIZE, vgg16.preprocess_input),
        (vgg19.VGG19, params.VGG19_IMG_SIZE, vgg19.preprocess_input),
        (mobilenet_v2.MobileNetV2, params.MOBILENETV2_IMG_SIZE, mobilenet_v2.preprocess_input),
        (inception_v3.InceptionV3, params.INCEPTIONV3_IMG_SIZE, inception_v3.preprocess_input),
        (nasnet.NASNetMobile, params.NASNETMOBILE_IMG_SIZE, nasnet.preprocess_input),
        (inception_resnet_v2.InceptionResNetV2, params.INCEPTIONRESNETV2_IMG_SIZE, inception_resnet_v2.preprocess_input)
    ]
    
    # list of custom models
    custom_model_name = ['simple', 'attention']

    # set optimizer
    gradient_accumulation_optimizer = gradient_accumulation.AdamAccumulate(lr=params.LEARNING_RATE, accum_iters=params.ACCUMULATION_STEPS)  # for these image sizes, we don't need gradient_accumulation to achieve BATCH_SIZE = 256
    optimizer = gradient_accumulation_optimizer #'adam'  # or gradient_accumulation_optimizer

    for b in base_models:
      for c in custom_model_name:
        # configure CNN model
        model = CNN_Model(transfer_model = b[0],
                          image_size = b[1],
                          preprocess_input = b[2],
                          custom_model_name = c,
                          fc_output = len(disease_classes),
                          optimizer = optimizer)
    
         # data generator
        train_generator = create_data_generator(train, disease_classes, params.BATCH_SIZE, target_size=b[1])
        valid_generator = create_data_generator(valid, disease_classes, params.BATCH_SIZE, target_size = b[1])
        #test_generator = create_data_generator(test, disease_classes, params.TEST_BATCH_SIZE, target_size=b[1])
    
        results_folder = os.path.join(params.RESULTS_FOLDER, model.model_name)
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
    
        callbacks_list = configure_callbacks(results_folder, model.model_name)
    
        # train the model
        history = model.final_model.fit_generator(generator=train_generator,
                                                  validation_data=valid_generator,
                                                  validation_steps=valid_generator.samples//valid_generator.batch_size,
                                                  steps_per_epoch=params.STEPS_PER_EPOCH,
                                                  epochs=params.EPOCHS,
                                                  callbacks=callbacks_list,
                                                  use_multiprocessing=True,
                                                  workers=params.WORKERS)
    
        RUN_TIMESTAMP = datetime.datetime.now().isoformat('-')
        # save loss and accuracy plots to disk
        plot_train_metrics(history, model.model_name, results_folder,  RUN_TIMESTAMP)
        # save json model config file and trained weights to disk
        save_model(model.final_model, model.model_name, results_folder, RUN_TIMESTAMP)
        # print ROC
        test_generator_plot = create_data_generator(test, disease_classes, batch_size=params.PLOT_BATCH_SIZE, target_size=b[1])
        test_X, test_Y = next(test_generator_plot)
    
        pred_Y = model.final_model.predict(test_X, batch_size=32, verbose=True)
        plot_ROC(disease_classes, test_Y, pred_Y, model.model_name)

if __name__ == '__main__':

  reset.reset_keras()
  run_procedure()
