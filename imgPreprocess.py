import pandas as pd
import numpy as np
import os
import random
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from scipy.io import loadmat

male_directory = 'c:/Users/Matt/Desktop/481/rendered_rgb/male/front/blurryBG'
female_directory = 'c:/Users/Matt/Desktop/481/rendered_rgb/female/front/blurryBG'

male_files = os.listdir(male_directory)
female_files = os.listdir(female_directory)

male_df = pd.DataFrame(male_files, columns=['filename'])

female_df = pd.DataFrame(female_files, columns=['filename'])

all_files = male_files + female_files
df = pd.DataFrame(all_files, columns=['filename'])

# Display the combined DataFrame
print(df.head())

# Optionally, save the DataFrame to a CSV file
df.to_csv('all_images.csv', index=False)

########################################################################################################################

mat_directory = 'c:/Users/Matt/Desktop/481/bodymeasurement_mat/male'

# List to store the DataFrames for each file
df_list = []

# Loop through all files in the directory
for filename in os.listdir(mat_directory):
    # Check if the file is a .mat file
    if filename.endswith('.mat'):
        mat_file_path = os.path.join(mat_directory, filename)

        # Load the .mat file
        data = loadmat(mat_file_path)
        
        # Access the structured array from the .mat file
        structured_array = data['s']
        
        # Get the field names
        field_names = structured_array.dtype.names
        
        # Dictionary to hold the paired data for this file
        paired_data = {}
        
        # Loop through each field in the structured array and extract the data
        for field in field_names:
            paired_data[field] = np.squeeze(structured_array[field][0]).item()

        # Create a DataFrame for this file and append it to the list
        df = pd.DataFrame([paired_data])
        df = df.applymap(lambda x: x if isinstance(x, (int, float)) else x.item())  # Ensure scalar values
        df['id'] = os.path.splitext(filename)[0]
        df_list.append(df)

# Concatenate all DataFrames into a single DataFrame
final_df = pd.concat(df_list, ignore_index=True)

print(final_df.head())

final_df.to_csv('male_measurements.csv', index=False)


########################################################################################################################

# datagen = ImageDataGenerator(
#         rotation_range=180,
#         width_shift_range=0.2,
#         height_shift_range=0.2,
#         rescale=1./255,
#         horizontal_flip=True,
#         vertical_flip=True,
#         fill_mode='nearest')

# front_train = 'BodyM/train/mask'
# side_train = 'BodyM/train/mask_left'

# train_measurements = pd.read_csv('BodyM/train/measurements.csv')

# train_map = pd.read_csv('BodyM/train/subject_to_photo_map.csv')
# train_map = train_map.set_index('photo_id')['subject_id'].to_dict()

# def multi_input_generator(front_filenames, side_filenames, labels, batch_size, image_dir_front, image_dir_side, target_size=(150, 150)):
#     while True:
#         for i in range(0, len(front_filenames), batch_size):
#             # Get batch filenames
#             front_batch = front_filenames[i:i + batch_size]
#             side_batch = side_filenames[i:i + batch_size]
#             batch_labels = labels[i:i + batch_size]
            
#             # Load and process images
#             front_images = []
#             side_images = []
#             for front_file, side_file in zip(front_batch, side_batch):
#                 # Load and preprocess front image
#                 front_img = load_img(os.path.join(image_dir_front, front_file), target_size=target_size)
#                 front_img = img_to_array(front_img) / 255.0  # Normalize
#                 front_images.append(front_img)
                
#                 # Load and preprocess side image
#                 side_img = load_img(os.path.join(image_dir_side, side_file), target_size=target_size)
#                 side_img = img_to_array(side_img) / 255.0  # Normalize
#                 side_images.append(side_img)
            
#             # Yield the images and labels
#             yield [np.array(front_images), np.array(side_images)], np.array(batch_labels)

# # train_generator = multi_input_generator(
# #     front_filenames=front_train,
# #     side_filenames=side_train,
# #     labels=labels_train,
# #     batch_size=32,
# #     image_dir_front='data/front/',
# #     image_dir_side='data/side/'
# # )

# # # Validation generator
# # val_generator = multi_input_generator(
# #     front_filenames=front_val,
# #     side_filenames=side_val,
# #     labels=labels_val,
# #     batch_size=32,
# #     image_dir_front='data/front/',
# #     image_dir_side='data/side/'
# # )

