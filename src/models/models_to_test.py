from keras import Model
from keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D, RandomFlip 
from keras.layers import RandomRotation, RandomZoom, RandomContrast
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.applications import EfficientNetV2B0, efficientnet_v2
from keras.optimizers import Adam

from src.models.base_model import Base_model
import tensorflow as tf

class PatchExtractor(tf.keras.layers.Layer):
    
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]

        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID"
        )

        # patches shape:
        # (batch, n_h, n_w, patch_size * patch_size * 3)

        n_h = tf.shape(patches)[1]
        n_w = tf.shape(patches)[2]

        # reshape 
        patches = tf.reshape(
            patches,
            (
                batch_size,
                n_h * n_w,
                self.patch_size,
                self.patch_size,
                3
            )
        )
        
        # Masque pour garder seulement les patchs qui ont au moins un pixel non nul
        non_empty_mask = tf.reduce_any(patches > 0, axis=[2,3,4])
        filtered_patches = tf.ragged.boolean_mask(patches, non_empty_mask)

        # Convertir en tensor dense (patchs vides complétés avec 0)
        filtered_patches = filtered_patches.to_tensor(default_value=0)

        return filtered_patches
    
    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config

class Base(Base_model):
    
    
    def __init__(self,
                 data_folder,
                 save_model_folder,
                 model_name,
                 img_size=(299, 299),
                 gray=False,
                 batch_size=32,
                 big_dataset=False,
                 test_size=0.2,
                 random_state=42,
                 oversampling=False):
        super().__init__(data_folder, save_model_folder, model_name, img_size, gray, batch_size, big_dataset, test_size, random_state, oversampling)
        
    def build_model(self):
        pass


class EfficientNetv2B0_model_augmented(Base_model):

        def __init__(self,
                    data_folder,
                    save_model_folder,
                    model_name,
                    img_size=(299, 299),
                    gray=False,
                    batch_size=32,
                    big_dataset=False,
                    test_size=0.2,
                    random_state=42,
                    oversampling=False,
                    class_weight=None,
                    nb_layer_to_freeze=0):
            
            self.nb_layer_to_freeze = nb_layer_to_freeze
            super().__init__(data_folder, save_model_folder, model_name, img_size, gray, batch_size, big_dataset, test_size, random_state, oversampling, class_weight)
            
            early_stopping = EarlyStopping(
                                    patience=5, 
                                    min_delta=1, 
                                    mode='min',
                                    monitor='loss')

            reduce_learning_rate = ReduceLROnPlateau(
                                        monitor="loss",
                                        patience=3,
                                        min_delta=0.01,
                                        factor=0.1, 
                                        cooldown=4)
            
            self.callbacks = [early_stopping, reduce_learning_rate]
            
        def build_model(self):
            # Entrée (3 canaux car Base_model charge en RGB automatiquement)
            inputs = Input(shape=(self.img_size[0], self.img_size[1], 3))
                
            # Data augmentation en layer
            x = RandomFlip("horizontal")(inputs)
            x = RandomRotation(0.1)(x)     
            x = RandomZoom(0.1)(x)         
            x = RandomContrast(0.1)(x)     

            x = efficientnet_v2.preprocess_input(x)

            # EfficientNetV2B0 pré-entraîné
            base = EfficientNetV2B0(
                include_top=False,
                weights='imagenet',
                input_tensor=x
            )
            
            for layer in base.layers[:self.nb_layer_to_freeze]:
                layer.trainable = False

            for layer in base.layers[self.nb_layer_to_freeze:]:
                layer.trainable = True
            
            x = GlobalAveragePooling2D()(base.output)
            x = Dense(256, activation='relu')(x)
            x = Dropout(0.5)(x)

            
            # Classifier
            outputs = Dense(2, activation='softmax')(x)


            model = Model(inputs=inputs, outputs=outputs)


            model.compile(
                optimizer=str(Adam(learning_rate=1e-4)),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )


            return model
        