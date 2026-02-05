from keras import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Rescaling, Dropout, GlobalAveragePooling2D, RandomFlip 
from keras.layers import RandomRotation, RandomZoom, RandomContrast, Resizing, TimeDistributed, GlobalAveragePooling1D
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.applications import DenseNet121, DenseNet201, EfficientNetB5, EfficientNetV2B0, EfficientNetV2B1
from keras.applications import EfficientNetV2B2, EfficientNetB0, EfficientNetB2, EfficientNetV2B3, VGG19, densenet, efficientnet_v2
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


class CNN(Base_model):
    
    
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
        
        early_stopping = EarlyStopping(
                                patience=5, 
                                min_delta=0.01, 
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

        """        
        inputs = Input(shape=(self.img_size[0], self.img_size[1], 3))
        x = Conv2D(32, (3,3), activation='relu')(inputs)
        x = MaxPooling2D()(x)
        x = Conv2D(64, (3,3), activation='relu')(x)
        x = MaxPooling2D()(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        outputs = Dense(2, activation='softmax')(x)
        """
        
        inputs = Input(shape=(self.img_size[0], self.img_size[1], 3))
        
        # Data augmentation en layer
        x = RandomFlip("horizontal")(inputs)
        x = RandomRotation(0.1)(x)     
        x = RandomZoom(0.1)(x)         
        x = RandomContrast(0.1)(x) 
        
        x = Conv2D(32, (3,3), activation='relu')(x)
        x = MaxPooling2D()(x)
        x = Conv2D(64, (3,3), activation='relu')(x)
        x = MaxPooling2D()(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        outputs = Dense(2, activation='softmax')(x)
        
        # Création du modèle
        model = Model(inputs=inputs, outputs=outputs)

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model
    
    
class Le_Net(Base_model):
    
    
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
        
        early_stopping = EarlyStopping(
                                patience=5, 
                                min_delta=0.01, 
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

        """        
        inputs = Input(shape=(self.img_size[0], self.img_size[1], 3))
        x = Rescaling(1./255)(inputs)
        x = Conv2D(filters=30, kernel_size=(5,5), activation='relu', padding="valid")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(filters=16, kernel_size=(2,2), activation='relu', padding="valid")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.2)(x)
        x = Flatten()(x)
        x = Dense(units=128, activation="relu")(x)
        outputs = Dense(units=10, activation='softmax')(x)
        ])"""
        
        inputs = Input(shape=(self.img_size[0], self.img_size[1], 3))
        
        # Data augmentation en layer
        x = RandomFlip("horizontal")(inputs)
        x = RandomRotation(0.1)(x)     
        x = RandomZoom(0.1)(x)         
        x = RandomContrast(0.1)(x) 
        
        x = Rescaling(1./255)(x)
        x = Conv2D(filters=30, kernel_size=(5,5), activation='relu', padding="valid")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(filters=16, kernel_size=(2,2), activation='relu', padding="valid")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.2)(x)
        x = Flatten()(x)
        x = Dense(units=128, activation="relu")(x)
        outputs = Dense(units=2, activation='softmax')(x)
        
        # Création du modèle
        model = Model(inputs=inputs, outputs=outputs)

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model
    
class DenseNet121_model(Base_model):

    def __init__(self,
                 data_folder,
                 save_model_folder,
                 model_name,
                 img_size=(224, 224),
                 gray=False,
                 batch_size=32,
                 big_dataset=False,
                 test_size=0.2,
                 random_state=42,
                 oversampling=False):
        super().__init__(data_folder, save_model_folder, model_name, img_size, gray, batch_size, big_dataset, test_size, random_state, oversampling)
        
        early_stopping = EarlyStopping(
                                patience=5, 
                                min_delta=0.01, 
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


        # DenseNet121 pré-entraîné
        base = DenseNet121(
            include_top=False,
            weights='imagenet',
            input_tensor=inputs,
            pooling='avg'
        )
        
        #x = GlobalAveragePooling2D()(base.output)
        
        # Classifier
        outputs = Dense(2, activation='softmax')(base.output)


        model = Model(inputs=inputs, outputs=outputs)


        model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )


        return model
    
class DenseNet121_model_augmented(Base_model):

    def __init__(self,
                 data_folder,
                 save_model_folder,
                 model_name,
                 img_size=(224, 224),
                 gray=False,
                 batch_size=32,
                 big_dataset=False,
                 test_size=0.2,
                 random_state=42,
                 oversampling=False):
        super().__init__(data_folder, save_model_folder, model_name, img_size, gray, batch_size, big_dataset, test_size, random_state, oversampling)
        
        early_stopping = EarlyStopping(
                                patience=5, 
                                min_delta=0.01, 
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

        x = densenet.preprocess_input(x)

        # DenseNet121 pré-entraîné
        base = DenseNet121(
            include_top=False,
            weights='imagenet',
            input_tensor=x
        )
        
        x = GlobalAveragePooling2D()(base.output)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)

        
        # Classifier
        outputs = Dense(2, activation='softmax')(x)


        model = Model(inputs=inputs, outputs=outputs)


        model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )


        return model
    
class DenseNet201_model_augmented(Base_model):

    def __init__(self,
                 data_folder,
                 save_model_folder,
                 model_name,
                 img_size=(224, 224),
                 gray=False,
                 batch_size=32,
                 big_dataset=False,
                 test_size=0.2,
                 random_state=42,
                 oversampling=False):
        super().__init__(data_folder, save_model_folder, model_name, img_size, gray, batch_size, big_dataset, test_size, random_state, oversampling)
        
        early_stopping = EarlyStopping(
                                patience=5, 
                                min_delta=0.01, 
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
        
        # resizing
        x = Resizing(224,224)(inputs)
        
        # Data augmentation en layer
        x = RandomFlip("horizontal")(x)
        x = RandomRotation(0.1)(x)     
        x = RandomZoom(0.1)(x)         
        x = RandomContrast(0.1)(x)     

        # DenseNet201 pré-entraîné
        base = DenseNet201(
            include_top=False,
            weights='imagenet',
            input_tensor=x
        )
        
        # Geler tous les blocs sauf les 3e et 4e
        for layer in base.layers:
            name = layer.name
            if name.startswith(("conv1", "pool1",
                                "conv2", "pool2",
                                "conv3", "pool3")):
                layer.trainable = False
            else:
                layer.trainable = True
        
        x = GlobalAveragePooling2D()(base.output)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)

        
        # Classifier
        outputs = Dense(2, activation='softmax')(x)


        model = Model(inputs=inputs, outputs=outputs)


        model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )


        return model
   

# -----------------------------------
# EfficientNetB0
# -----------------------------------
class EfficientNetB0_Modele(Base_model):


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


        early_stopping = EarlyStopping(patience=5, 
                                       min_delta=0.01, 
                                       mode='min', 
                                       monitor='loss')
        
        reduce_learning_rate = ReduceLROnPlateau(monitor="loss", 
                                                 patience=3, 
                                                 min_delta=0.01, 
                                                 factor=0.1, 
                                                 cooldown=4)
        
        self.callbacks = [early_stopping, reduce_learning_rate]


    def build_model(self):
        
        inputs = Input(shape=(self.img_size[0], self.img_size[1], 3))
        
        base = EfficientNetB0(include_top=False, 
                              weights='imagenet', 
                              input_tensor=inputs, 
                              pooling='avg')
        
        outputs = Dense(2, activation='softmax')(base.output)


        model = Model(inputs=inputs, outputs=outputs)
        
        model.compile(optimizer=Adam(learning_rate=1e-4), 
                      loss='categorical_crossentropy', 
                      metrics=['accuracy'])
        
        return model

# -----------------------------------
# EfficientNetB2 (uniquement avec une trés bonne machine)
# -----------------------------------
class EfficientNetB2_Modele(Base_model):


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


        early_stopping = EarlyStopping(patience=5, 
                                       min_delta=0.01, 
                                       mode='min', 
                                       monitor='loss')
        
        reduce_learning_rate = ReduceLROnPlateau(monitor="loss", 
                                                 patience=3, 
                                                 min_delta=0.01, 
                                                 factor=0.1, 
                                                 cooldown=4)
        
        self.callbacks = [early_stopping, reduce_learning_rate]


    def build_model(self):
        
        inputs = Input(shape=(self.img_size[0], self.img_size[1], 3))
        
        # resizing
        x = Resizing(260,260)(inputs)
        
        # Data augmentation en layer
        x = RandomFlip("horizontal")(x)
        x = RandomRotation(0.1)(x)     
        x = RandomZoom(0.1)(x)         
        x = RandomContrast(0.1)(x)    
        
        base = EfficientNetB2(include_top=False, 
                              weights='imagenet', 
                              input_tensor=x)
        
        x = GlobalAveragePooling2D()(base.output)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        
        outputs = Dense(2, activation='softmax')(x)


        model = Model(inputs=inputs, outputs=outputs)
        
        model.compile(optimizer=Adam(learning_rate=1e-4), 
                      loss='categorical_crossentropy', 
                      metrics=['accuracy'])
        
        return model
    
# -----------------------------------
# EfficientNetB5 (uniquement avec une trés bonne machine)
# -----------------------------------
class EfficientNetB5_Modele_augmented(Base_model):


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


        early_stopping = EarlyStopping(patience=5, 
                                       min_delta=0.01, 
                                       mode='min', 
                                       monitor='loss')
        
        reduce_learning_rate = ReduceLROnPlateau(monitor="loss", 
                                                 patience=3, 
                                                 min_delta=0.01, 
                                                 factor=0.1, 
                                                 cooldown=4)
        
        self.callbacks = [early_stopping, reduce_learning_rate]


    def build_model(self):
        
        inputs = Input(shape=(self.img_size[0], self.img_size[1], 3))
                
        # Data augmentation en layer
        x = RandomFlip("horizontal")(inputs)
        x = RandomRotation(0.2)(x)     
        x = RandomZoom(0.2)(x)         
        x = RandomContrast(0.2)(x)
        
        x = efficientnet_v2.preprocess_input(x)
        
        base = EfficientNetB5(include_top=False, 
                              weights='imagenet', 
                              input_tensor=x)
        
        # Geler toute la base dans un premier temps
        base.trainable = True
        for layer in base.layers:
            layer.trainable = False

        # Ré-activer seulement les blocs MBConv6
        for layer in base.layers:
            if "block6" in layer.name:    
                layer.trainable = True
        
        x = GlobalAveragePooling2D()(base.output)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        
        outputs = Dense(2, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=outputs)
        
        model.compile(optimizer=Adam(learning_rate=1e-4), 
                      loss='categorical_crossentropy', 
                      metrics=['accuracy'])
        
        return model

class VGG19_model_augmented(Base_model):

    def __init__(self,
                 data_folder,
                 save_model_folder,
                 model_name,
                 img_size=(224, 224),
                 gray=False,
                 batch_size=32,
                 big_dataset=False,
                 test_size=0.2,
                 random_state=42,
                 oversampling=False):
        super().__init__(data_folder, save_model_folder, model_name, img_size, gray, batch_size, big_dataset, test_size, random_state, oversampling)
        
        early_stopping = EarlyStopping(
                                patience=5, 
                                min_delta=0.01, 
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
        
        # resizing
        x = Resizing(224,224)(inputs)
        
        # Data augmentation en layer
        x = RandomFlip("horizontal")(x)
        x = RandomRotation(0.1)(x)     
        x = RandomZoom(0.1)(x)         
        x = RandomContrast(0.1)(x)     

        # VGG19 pré-entraîné
        base = VGG19(
            include_top=False,
            weights='imagenet',
            input_tensor=x
        )
        
        # Geler blocs 1 à 3
        for layer in base.layers:
            if layer.name.startswith(("block1", "block2", "block3")):
                layer.trainable = False
            else:
                layer.trainable = True
        
        x = GlobalAveragePooling2D()(base.output)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)

        
        # Classifier
        outputs = Dense(2, activation='softmax')(x)


        model = Model(inputs=inputs, outputs=outputs)


        model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )


        return model

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
                                    min_delta=0.01, 
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
                optimizer=Adam(learning_rate=1e-4),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )


            return model
        
        
class EfficientNetv2B1_model_augmented(Base_model):

        def __init__(self,
                    data_folder,
                    save_model_folder,
                    model_name,
                    img_size=(240, 240),
                    gray=False,
                    batch_size=32,
                    big_dataset=False,
                    test_size=0.2,
                    random_state=42,
                    oversampling=False):
            super().__init__(data_folder, save_model_folder, model_name, img_size, gray, batch_size, big_dataset, test_size, random_state, oversampling)
            
            early_stopping = EarlyStopping(
                                    patience=5, 
                                    min_delta=0.01, 
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

            # EfficientNetV2B1 pré-entraîné
            base = EfficientNetV2B1(
                include_top=False,
                weights='imagenet',
                input_tensor=x
            )
            
            freeze_until = 20

            for layer in base.layers[:freeze_until]:
                layer.trainable = False

            for layer in base.layers[freeze_until:]:
                layer.trainable = True
            
            x = GlobalAveragePooling2D()(base.output)
            x = Dense(256, activation='relu')(x)
            x = Dropout(0.5)(x)

            
            # Classifier
            outputs = Dense(2, activation='softmax')(x)


            model = Model(inputs=inputs, outputs=outputs)


            model.compile(
                optimizer=Adam(learning_rate=1e-4),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )


            return model
        
class EfficientNetv2B2_model_augmented(Base_model):

        def __init__(self,
                    data_folder,
                    save_model_folder,
                    model_name,
                    img_size=(260, 260),
                    gray=False,
                    batch_size=32,
                    big_dataset=False,
                    test_size=0.2,
                    random_state=42,
                    oversampling=False):
            super().__init__(data_folder, save_model_folder, model_name, img_size, gray, batch_size, big_dataset, test_size, random_state, oversampling)
            
            early_stopping = EarlyStopping(
                                    patience=5, 
                                    min_delta=0.01, 
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

            # EfficientNetV2B2 pré-entraîné
            base = EfficientNetV2B2(
                include_top=False,
                weights='imagenet',
                input_tensor=x
            )
            
            freeze_until = 20

            for layer in base.layers[:freeze_until]:
                layer.trainable = False

            for layer in base.layers[freeze_until:]:
                layer.trainable = True
            
            x = GlobalAveragePooling2D()(base.output)
            x = Dense(256, activation='relu')(x)
            x = Dropout(0.5)(x)

            
            # Classifier
            outputs = Dense(2, activation='softmax')(x)


            model = Model(inputs=inputs, outputs=outputs)


            model.compile(
                optimizer=Adam(learning_rate=1e-4),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )


            return model
        
class EfficientNetv2B3_model_augmented(Base_model):

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
                    nb_layer_to_freeze=0):
            self.nb_layer_to_freeze = nb_layer_to_freeze
            super().__init__(data_folder, save_model_folder, model_name, img_size, gray, batch_size, big_dataset, test_size, random_state, oversampling)
            
            early_stopping = EarlyStopping(
                                    patience=5, 
                                    min_delta=0.01, 
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

            x = Rescaling(1./255)(x)

            # EfficientNetV2B3 pré-entraîné
            base = EfficientNetV2B3(
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
                optimizer=Adam(learning_rate=1e-4),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )


            return model
        
        
class EfficientNetv2B0_PatchCNN_model(Base_model):

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
        
        super().__init__(
            data_folder,
            save_model_folder,
            model_name,
            img_size,
            gray,
            batch_size,
            big_dataset,
            test_size,
            random_state,
            oversampling
        )

        early_stopping = EarlyStopping(
            patience=5,
            min_delta=0.01,
            mode="min",
            monitor="loss"
        )

        reduce_learning_rate = ReduceLROnPlateau(
            monitor="loss",
            patience=3,
            min_delta=0.01,
            factor=0.1,
            cooldown=4
        )

        self.callbacks = [early_stopping, reduce_learning_rate]

    def build_model(self):

        # ===== Input =====
        inputs = Input(shape=(self.img_size[0], self.img_size[1], 3))

        # ===== Data augmentation =====
        x = RandomFlip("horizontal")(inputs)
        x = RandomRotation(0.1)(x)
        x = RandomZoom(0.1)(x)
        x = RandomContrast(0.1)(x)

        # ===== EfficientNetV2 preprocessing =====
        x = Rescaling(1./255)(x)

        # ===== Patch extraction =====
        patches = PatchExtractor(patch_size=96)(x)

        # ===== Backbone =====
        backbone = EfficientNetV2B0(
            include_top=False,
            weights="imagenet",
            input_shape=(96, 96, 3)
        )

        features = TimeDistributed(backbone)(patches)
        features = TimeDistributed(GlobalAveragePooling2D())(features)
        features = GlobalAveragePooling1D()(features)

        # ===== Classifier =====
        x = Dense(256, activation="relu")(features)
        x = Dropout(0.5)(x)
        outputs = Dense(2, activation="softmax")(x)

        model = Model(inputs=inputs, outputs=outputs)

        model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        return model
