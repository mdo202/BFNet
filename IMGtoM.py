import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB5
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping

class IMGtoM:

    def collect_paired_files(category):
        base_dir = "Data/rendered_rgb"
        genders = ["male", "female"]
        paired_files = []
        
        for gender in genders:
            front_dir = os.path.join(base_dir, gender, "front", category)
            side_dir = os.path.join(base_dir, gender, "side", category)
            if os.path.exists(front_dir) and os.path.exists(side_dir):
                front_files = set(os.listdir(front_dir))
                side_files = set(os.listdir(side_dir))
                common_files = front_files.intersection(side_files)
                for filename in common_files:
                    paired_files.append({
                        "front_image": os.path.join(front_dir, filename),
                        "side_image": os.path.join(side_dir, filename),
                        "gender": gender,
                        "category": category,
                        "ID": filename[:-4]
                    })
        return paired_files

    def get_data():
        categories = ["blurryBG", "clearBG", "noiseBG", "silh", "woBG"]
        data = []
        
        for category in categories:
            data.extend(collect_paired_files(category))
            
        df = pd.DataFrame(data)
        df_train_val = df[df["category"] != "woBG"]     # noise + sihlouettes
        df_train, df_val = train_test_split(df_train_val, test_size=0.2, stratify=df_train_val["category"]) 
        df_test = df[df["category"] == "woBG"]      # actual images
        
        df_f_m = pd.read_csv("Data/female_measurements_cleaned.csv")
        df_m_m = pd.read_csv("Data/male_measurements_cleaned.csv")
        
        female_ids = df_f_m["id"].tolist()
        male_ids = df_m_m["id"].tolist()
        
        df_train = df_train[df_train["ID"].isin(female_ids + male_ids)]
        df_val = df_val[df_val["ID"].isin(female_ids + male_ids)]
        df_test = df_test[df_test["ID"].isin(female_ids + male_ids)]   
        
        return df_train, df_val, df_test, df_f_m, df_m_m

    '''
    MtoBF features mapped to IMGtoM features:
        -Neck
            Male:   Neck_Circ
            Female: Neck_Circ
        -Chest
            Male: CHEST_Circ
            Female: BUST_Circ
        -Abdomen
            Male: MaxWAIST_Circ
            Female: NaturalWAIST_Circ (?????????????)
        -Hip
            Male:   HIP_Circ
            Female: HIP_Circ
        -Thigh
            Male:   Thigh_Circ
            Female: Thigh_Circ
        -Knee
            Male:   Knee_Circ
            Female: Knee_Circ
        -Ankle
            Male:   Ankle_Circ
            Female: Ankle_Circ
        -Biceps
            Male:   Bicep_Circ
            Female: Bicep_Circ
        -Forearm
            Male: Elbow_Circ (???)
            Female: Elbow_Circ (???)
        -Wrist  
            Male: Wrist_Circ
            Female: Wrist_Circ
    '''

    def find_labels(ids, genders, task, df_m_m, df_f_m):
        
        labels = []
        
        for idx, id in enumerate(ids):

            if genders[idx] == 'male':
                
                if task == "Neck":
                    labels.append(df_m_m.loc[df_m_m["id"] == id, "Neck_Circ"].item())
                elif task == "Chest":
                    labels.append(df_m_m.loc[df_m_m["id"] == id, "CHEST_Circ"].item())
                elif task == "Abdomen":
                    labels.append(df_m_m.loc[df_m_m["id"] == id, "MaxWAIST_Circ"].item()) ### Using max waist
                elif task == "Hip":
                    labels.append(df_m_m.loc[df_m_m["id"] == id, "HIP_Circ"].item())
                elif task == "Thigh":
                    labels.append(df_m_m.loc[df_m_m["id"] == id, "Thigh_Circ"].item())
                elif task == "Knee":
                    labels.append(df_m_m.loc[df_m_m["id"] == id, "Knee_Circ"].item())
                elif task == "Ankle":
                    labels.append(df_m_m.loc[df_m_m["id"] == id, "Ankle_Circ"].item())
                elif task == "Biceps":
                    labels.append(df_m_m.loc[df_m_m["id"] == id, "Bicep_Circ"].item())
                elif task == "Forearm":
                    labels.append(df_m_m.loc[df_m_m["id"] == id, "Elbow_Circ"].item()) ### ??????????????????????
                elif task == "Wrist":
                    labels.append(df_m_m.loc[df_m_m["id"] == id, "Wrist_Circ"].item())
                else:
                    raise ValueError("Task Not Found")
                
            elif genders[idx] == 'female':
                
                if task == "Neck":
                    labels.append(df_f_m.loc[df_f_m["id"] == id, "Neck_Circ"].item())
                elif task == "Chest":
                    labels.append(df_f_m.loc[df_f_m["id"] == id, "BUST_Circ"].item()) ### Using bust circ
                elif task == "Abdomen":
                    labels.append(df_f_m.loc[df_f_m["id"] == id, "NaturalWAIST_Circ"].item()) ### Using natural waist
                elif task == "Hip":
                    labels.append(df_f_m.loc[df_f_m["id"] == id, "HIP_Circ"].item())
                elif task == "Thigh":
                    labels.append(df_f_m.loc[df_f_m["id"] == id, "Thigh_Circ"].item())
                elif task == "Knee":
                    labels.append(df_f_m.loc[df_f_m["id"] == id, "Knee_Circ"].item())
                elif task == "Ankle":
                    labels.append(df_f_m.loc[df_f_m["id"] == id, "Ankle_Circ"].item())
                elif task == "Biceps":
                    labels.append(df_f_m.loc[df_f_m["id"] == id, "Bicep_Circ"].item())
                elif task == "Forearm":
                    labels.append(df_f_m.loc[df_f_m["id"] == id, "Elbow_Circ"].item()) ### ??????????????????????
                elif task == "Wrist":
                    labels.append(df_f_m.loc[df_f_m["id"] == id, "Wrist_Circ"].item())
                else:
                    raise ValueError("Task Not Found")
                
        return labels
        

    def multi_input_data_generator(df, task, df_m_m, df_f_m, batch_size=16, target_size=(456, 456), shuffle=True, augment=True):
        
        front_images = df["front_image"].values
        side_images = df["side_image"].values
        ids = df["ID"].values
        genders = df["gender"].values
        labels = np.array(find_labels(ids, genders, task, df_m_m, df_f_m)) / 10 ### measurements are in mm, BodyFat Extended measurements in cm 

        if shuffle:
            indices = np.arange(len(front_images))
            np.random.shuffle(indices)
            front_images = front_images[indices]
            side_images = side_images[indices]
            labels = labels[indices]
            
        if augment:
            datagen = ImageDataGenerator(
                rotation_range=180,
                width_shift_range=0.1,
                height_shift_range=0.1,
                rescale=1./255,
                horizontal_flip=True,
                vertical_flip=True,
                fill_mode='nearest'
            )
        else:
            datagen = ImageDataGenerator(rescale=1./255) 
            
        while True:
            
            for i in range(0, len(front_images), batch_size):
                front_batch = front_images[i:i + batch_size]
                side_batch = side_images[i:i + batch_size]
                batch_labels = labels[i:i + batch_size]
                
                front_batch_images = []
                side_batch_images = []
                
                for front_path, side_path in zip(front_batch, side_batch):
                    front_img = load_img(front_path, target_size=target_size)
                    front_img = img_to_array(front_img)
                    front_img = datagen.random_transform(front_img)  
                    front_img = datagen.standardize(front_img)  
                    front_batch_images.append(front_img)
                    
                    side_img = load_img(side_path, target_size=target_size)
                    side_img = img_to_array(side_img) 
                    side_img = datagen.random_transform(side_img) 
                    side_img = datagen.standardize(side_img) 
                    side_batch_images.append(side_img)
                
                front_batch_images = tf.convert_to_tensor(np.array(front_batch_images), dtype=tf.float32)
                side_batch_images = tf.convert_to_tensor(np.array(side_batch_images), dtype=tf.float32)
                batch_labels = tf.convert_to_tensor(np.array(batch_labels), dtype=tf.float32)
                
                yield ((front_batch_images), (side_batch_images)), batch_labels
                
    #####################################################################################################################################

    def train(task, save=False):

        df_train, df_val, df_test, df_f_m, df_m_m = get_data()

        # Build the model
        front_input = Input(shape=(456, 456, 3), name="front_input")
        side_input = Input(shape=(456, 456, 3), name="side_input")

        front_base = EfficientNetB5(include_top=False, weights="imagenet", input_shape=(456, 456, 3))
        front_base._name = "EfficientNetB5_front"
        front_features = front_base(front_input)
        front_flatten = Flatten()(front_features)

        side_base = EfficientNetB5(include_top=False, weights="imagenet", input_shape=(456, 456, 3))
        side_base._name = "EfficientNetB5_side"
        side_features = side_base(side_input)
        side_flatten = Flatten()(side_features)

    ### Regularize entire model

        # for layer in front_base.layers:
        #     layer.kernel_regularizer = regularizers.l2(0.01)

        # for layer in side_base.layers:
        #     layer.kernel_regularizer = regularizers.l2(0.01)

    ### Regularize last layer

        # front_base.layers[-1].kernel_regularizer = regularizers.l2(0.01)
        # side_base.layers[-1].kernel_regularizer = regularizers.l2(0.01)
            
        merged = Concatenate()([front_flatten, side_flatten])
        dense_1 = Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.01))(merged)
        output = Dense(1, activation="linear", name="output")(dense_1)

        model = Model(inputs=[front_input, side_input], outputs=output)

        model.summary()

        for layer in front_base.layers:
            layer.trainable = False
        for layer in side_base.layers:
            layer.trainable = False

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                    loss="mean_squared_error", metrics=["mae", "mse"])

        early_stopping = EarlyStopping(
            monitor="val_loss",  # Metric to monitor
            patience=5,          # Number of epochs to wait for improvement
            restore_best_weights=True  # Restore best weights after stopping
        )

        with tf.device('/GPU:0'):
            # Train the model
            train_generator = multi_input_data_generator(df_train, task=task, df_m_m=df_m_m, df_f_m=df_f_m, batch_size=16, augment=True)
            val_generator = multi_input_data_generator(df_val, task=task, df_m_m=df_m_m, df_f_m=df_f_m, batch_size=16, augment=False)

            train_steps = len(df_train) // 16
            val_steps = len(df_val) // 16

            history = model.fit(train_generator, 
                                steps_per_epoch=train_steps,
                                validation_data=val_generator, 
                                validation_steps=val_steps, 
                                epochs=1000,
                                callbacks=[early_stopping])

    ### Fine tune the entire model
        
        # with tf.device('/GPU:0'):
        #     # Fine-tune the model
        #     for layer in front_base.layers:
        #         layer.trainable = True
        #     for layer in side_base.layers:
        #         layer.trainable = True

    ### Fine tune last layer

        # with tf.device('/GPU:0'):
        #     for layer in front_base.layers[:-1]:
        #         layer.trainable = False
        #     front_base.layers[-1].trainable = True

        #     for layer in side_base.layers[:-1]:
        #         layer.trainable = False
        #     side_base.layers[-1].trainable = True

        #     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        #                 loss="mean_squared_error", metrics=["mae", "mse"])

        #     history_fine_tune = model.fit(train_generator, 
        #                                 steps_per_epoch=train_steps,
        #                                 validation_data=val_generator, 
        #                                 validation_steps=val_steps, 
        #                                 epochs=1000,
        #                                 callbacks=[early_stopping])

        with tf.device('/GPU:0'):
            # Evaluate the model
            test_generator = multi_input_data_generator(df_test, task=task, df_m_m=df_m_m, df_f_m=df_f_m, batch_size=16, augment=False)
            test_steps = len(df_test) // 16
            test_loss, test_mae, test_mse = model.evaluate(test_generator, steps=test_steps)

        print(f"Test MAE: {test_mae}, Test MSE: {test_mse}")
        
        if save:
            save_name = f"IMGtoM_{task}.keras"
            model.save_weights(save_name)
        else:
            print("---Save Flag is False---")    

    '''Test grabs all the measurements & returns them as a list'''

    def test():
        
        # Build the model
        front_input = Input(shape=(456, 456, 3), name="front_input")
        side_input = Input(shape=(456, 456, 3), name="side_input")

        front_base = EfficientNetB5(include_top=False, weights="imagenet", input_shape=(456, 456, 3))
        front_base._name = "EfficientNetB5_front"
        front_features = front_base(front_input)
        front_flatten = Flatten()(front_features)

        side_base = EfficientNetB5(include_top=False, weights="imagenet", input_shape=(456, 456, 3))
        side_base._name = "EfficientNetB5_side"
        side_features = side_base(side_input)
        side_flatten = Flatten()(side_features)


        merged = Concatenate()([front_flatten, side_flatten])
        dense_1 = Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.01))(merged)
        output = Dense(1, activation="linear", name="output")(dense_1)

        model = Model(inputs=[front_input, side_input], outputs=output)

        model.summary()

        for layer in front_base.layers:
            layer.trainable = False
        for layer in side_base.layers:
            layer.trainable = False

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                    loss="mean_squared_error", metrics=["mae", "mse"])
        
        tasks = ["Neck", "Chest", "Abdomen", "Hip", "Thigh", "Knee", "Ankle", "Biceps", "Forearm", "Wrist"]
        predictions = {}
        
        for task in tasks:
            
            model.load_weights(f'Models/IMGtoM_{task}.keras')
        
            front_image_path = 'Test/test_front.png'
            front_image = load_img(front_image_path, target_size=(456, 456))  # Match input shape
            front_image = img_to_array(front_image)
            front_image = front_image / 255.0  # Normalize the image
            front_image = np.expand_dims(front_image, axis=0)
            
            side_image_path = 'Test/test_side.png'
            side_image = load_img(side_image_path, target_size=(456, 456))  # Match input shape
            side_image = img_to_array(side_image)
            side_image = side_image / 255.0  # Normalize the image
            side_image = np.expand_dims(side_image, axis=0)
            
            prediction = model.predict([front_image, side_image])
            print(f"Predicted {task} Circumfirence, in cm: {prediction[0][0]}")
            
            predictions[task] = prediction[0][0]
        
        predictions_df = pd.DataFrame([predictions])
                
        return predictions_df 

if __name__ == "__main__":
    
#     tasks = ["Neck", "Chest", "Abdomen", "Hip", "Thigh", "Knee", "Ankle", "Biceps", "Forearm", "Wrist"]
#     task = "Wrist" 
#     #Completed: Neck, Chest, Abdomen, Hip, Thigh, Knee, Ankle, Biceps, Forearm, Wrist
    
#     IMGtoM.train(task, save=False)

    p = IMGtoM.test()
    print(p)