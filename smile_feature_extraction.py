# libraries

import cv2
import mediapipe as mp
import os
import sys
from tqdm import tqdm
import csv
from protobuf_to_dict import protobuf_to_dict
import shutil
import pandas as pd
import numpy as np
from scipy.stats import entropy as ent
import re
import pickle
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, classification_report, confusion_matrix, roc_curve, auc
import argparse


# Function to extract the participant ID, task name, and date of the data sample
# Update this code to extract these properties based on your dataset
def extract_ID_task_name(file_name):
    if file_name[0].isdigit():
        first_underscore_index = file_name.find('_')
        if first_underscore_index != -1:
            ID = file_name[first_underscore_index+1:file_name.find('_', first_underscore_index+1)]
            task_name = file_name[file_name.find('_', first_underscore_index+1)+1:file_name.find('.', file_name.find('_', first_underscore_index+1))]
        else:
            task_name = file_name[file_name.rfind('-')+1:file_name.rfind('.')]
            ID = file_name[file_name.rfind('-', 0, file_name.rfind('-'))+1:file_name.rfind('-')]
    else:
        first_hyphen_index = file_name.find('-')
        ID = file_name[:first_hyphen_index]
        second_hyphen_index = file_name.find('-', first_hyphen_index+1)
        task_name = file_name[first_hyphen_index+1:second_hyphen_index]
    
    date = re.search(r'\d{4}-\d{2}-\d{2}', file_name).group()
    return file_name, task_name, date 



# function to generate a dictionary to store the intermediate content relavant to the features
# in the metadata, there will be 
# information for necessary task naming, 
# the Action Units for different tasks, 
# the facial landmarks for each of the landmark features,
# the raw feature data
def generate_metadata_and_data_dictionary(files_directory):
    dictionary = {
        'metadata':
        {
            'task_names': ['smile', 'disgust', 'surprise'],
            'task_numbers': ['task12', 'task13', 'task14'],
            'task_names_numbers_to_names': {
                'task12': 'smile',
                'task13': 'disgust',
                'task14': 'surprise',
                'smile': 'smile',
                'disgust': 'disgust',
                'surprise': 'surprise'
            },
            'tasks_required': ['smile'],
            "facial_features": {
                'eye-open-right': ['246', '7', '161', '163', '160', '144', '159', '145', '158', '153', '157', '154', '173', '155', '7'],
                'eye-open-left': ['398', '382', '384', '381', '385', '380', '386', '374', '387', '373', '388', '390', '466', '249', '7'],
                'eye-raise-right': ["246", '156', '161', '70', '160', '63', '159', '105', '158', '66', '157', '107', '173', '9', '7'],
                'eye-raise-left': ['398', '9', '384', '336', '385', '296', '386', '334', '387', '293', '388', '300', '466', '383', '7'],
                'mouth-open': ['191', '95', '80', '88', '81', '178', '82', '87', '13', '14', '312', '317', '311', '402', '310', '318', '415', '324', '9'],
                'mouth-width': ['78', '308', '1'],
                'jaw-open': ['191', '169', '80', '170', '81', '145', '82', '171', '13', '175', '312', '396', '311', '389', '310', '395', '415', '394', '9'],
            },
            'AU_values': {
                'smile': [1, 6, 12, 14, 25, 26, 45],
                'disgust': [4, 7, 9, 10, 25, 26, 45],
                'surprise': [1, 2, 4, 5, 25, 26, 45]
            }
        },
        'data':
        {

        }
    }

    # iterate through each of the files in the input directory
    # if the file is a video file then extract the ID and task name

    for file in os.listdir(files_directory):
        # check if the file is a video file
        # the video files are for the landmark extraction and the csv files are for openface action units
        if file.endswith('.mp4') or file.endswith('.webm') or file.endswith('.csv'):
            # extract the ID and task name
            ID, task_name, date = extract_ID_task_name(file)
            file_name = os.path.join(files_directory, file)
            # check if the ID is in the dictionary
            if ID not in dictionary["data"]:
                # add the ID to the dictionary
                dictionary['data'][ID] = {
                    date: {
                        dictionary['metadata']['task_names_numbers_to_names'][task_name] + '_file': file_name
                    }
                }
            else:
                # check if the date is in the dictionary
                if date not in dictionary['data'][ID]:
                    # add the date to the dictionary
                    dictionary['data'][ID][date] = {
                        dictionary['metadata']['task_names_numbers_to_names'][task_name] + '_file': file_name
                    }
                else:
                    # check if the task name is in the dictionary
                    if dictionary['metadata']['task_names_numbers_to_names'][task_name] + '_file' not in dictionary['data'][ID][date]:
                        # add the task name to the dictionary
                        dictionary['data'][ID][date][dictionary['metadata']['task_names_numbers_to_names'][task_name] + '_file'] = file_name
                    else:
                        # append the file to the list
                        #dictionary['data'][ID][date][dictionary['metadata']['task_names_numbers_to_names'][task_name] + '_file'].append(file_name)
                        print(file_name)
                        print('The ID and date combination already exists in the dictionary')
    
    dictionary_new = {
        'metadata': dictionary['metadata'],
        'data':
        {

        }
    }

    # for each ID, for each date, check if all the tasks are present
    # if not, then remove the ID from the dictionary
    for ID in dictionary['data']:
        flag = True
        for task in dictionary['metadata']['tasks_required']:
            for date in dictionary['data'][ID]:
                if task + '_file' not in dictionary['data'][ID][date]:
                    flag = False
        if flag:
            dictionary_new['data'][ID] = dictionary['data'][ID]

    del dictionary

    dictionary = dictionary_new
    
    return dictionary


# function that will read the single input video file and apply mediapipe facemesh to extract the landmark features
def extract_mediapipe_features(video_file, dictionary):
    # create a dataframe that will contain the mediapipe features
    columns = ['frame_number']
    for i in range(478):
        columns.append('x' + str(i))
        columns.append('y' + str(i))
        columns.append('z' + str(i))
    mediapipe_features_df = pd.DataFrame(columns=columns)
    feature_dict = {}
    # read the video as a video stream
    videocap = cv2.VideoCapture(video_file)
    # read the first frame
    success, image = videocap.read()
    # initialize the frame count
    count = 0
    # iterate through all the frames in the video
    while success:
        # extract facial landmarks from the frame
        mp_drawing = mp.solutions.drawing_utils
        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7) as face_mesh:
            # Convert the BGR image to RGB and process it with MediaPipe Face Mesh.
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))                        
            if results.multi_face_landmarks:
                frame_data = [count]
                for face_landmarks in results.multi_face_landmarks:
                    keypoints = protobuf_to_dict(face_landmarks)
                    for i in range(478):
                        point = keypoints['landmark'][i]
                        frame_data.append(point['x'])
                        frame_data.append(point['y'])
                        frame_data.append(point['z'])
                feature_dict[count] = frame_data

        # read the next frame
        success, image = videocap.read()
        # increment the frame count
        count += 1
        # print(count)
    # prepare the dataframe from the feature_dict
    for key in feature_dict:
        mediapipe_features_df.loc[len(mediapipe_features_df.index)] = feature_dict[key]
        # mediapipe_features_df = pd.concat([mediapipe_features_df, feature_dict[key]], ignore_index=True)

    facial_feature_values = {}
    for key in dictionary['metadata']['facial_features']:
        facial_feature_values[key] = []

    for inedx, row in mediapipe_features_df.iterrows():
        row = row.to_dict()
        # find the distance between the two centroids of the eyes
        eye_distance = np.sqrt((float(row['x468']) - float(row['x473']))**2 + (float(row['y468']) - float(row['y473']))**2)
        # print(eye_distance)

        for facial_feature in dictionary['metadata']['facial_features']:
            facial_feature_sum = 0
            for i in range(0, len(dictionary['metadata']['facial_features'][facial_feature])-2, 2):
                try:
                    facial_feature_sum += np.sqrt((float(row['x'+dictionary['metadata']['facial_features'][facial_feature][i]]) - float(row['x'+dictionary['metadata']['facial_features'][facial_feature][i+1]]))**2 + (float(row['y'+dictionary['metadata']['facial_features'][facial_feature][i]]) - float(row['y'+dictionary['metadata']['facial_features'][facial_feature][i+1]]))**2)
                except:
                    facial_feature_sum += 0
            facial_feature_sum /= float(dictionary['metadata']['facial_features'][facial_feature][-1])
            # if facial_feature == 'mouth-width':
                    # print(facial_feature_sum)
            facial_feature_values[facial_feature].append(facial_feature_sum/eye_distance)

    data_line = []
    # append the mean, variance, entropy for each of the facial features

    for facial_feature in dictionary['metadata']['facial_features']:
        if len(facial_feature_values[facial_feature]) > 0:
            data_line.append(np.mean(facial_feature_values[facial_feature]))
            data_line.append(np.var(facial_feature_values[facial_feature]))
            data_line.append(ent(facial_feature_values[facial_feature]))
        else:
            data_line.append(0)
            data_line.append(0)
            data_line.append(0)
        
    return data_line

# function that reads all the videos from a directory and generate the landmark feature dataset
def create_mediapipe_features_dataframe(video_files_directory):
    
    dictionary = generate_metadata_and_data_dictionary(video_files_directory)

    for ID in tqdm(dictionary['data'], total=len(dictionary['data']), file=sys.stdout):
        for task in dictionary['metadata']['tasks_required']:
            for date in dictionary['data'][ID]:
                file_name = dictionary['data'][ID][date][task + '_file']
                mp_features = extract_mediapipe_features(os.path.join(file_name), dictionary)
                dictionary['data'][ID][date][task + '_mp_features'] = mp_features
                
    
    # we now have the mediapipe features for each of the videos
    # create a dataframe with columns as the userID, and mediapipe features
    columns = ['ID', 'date']

    for task in dictionary['metadata']['tasks_required']:
        for feature in dictionary['metadata']['facial_features']:
            columns.append(task + '_' + feature + '_mean')
            columns.append(task + '_' + feature + '_var')
            columns.append(task + '_' + feature + '_entropy')
            
    features_df = pd.DataFrame(columns=columns)

    # we now have the mediapipe features for each of the videos
    # for a particular user, if there are multiple videos and any of those vidoes videos fails to generate the mediapipe features, then we will not consider the mediapipe features for that user
    # in that case discard the mediapipe features for that user and discard the video file name from the user's data
    for ID in dictionary['data']:
        for date in dictionary['data'][ID]:
            flag = True
            for task in dictionary['metadata']['tasks_required']:
                current_mp_features = dictionary['data'][ID][date][task + '_mp_features']
                for i in range(0, len(current_mp_features), 3):
                    if current_mp_features[i] == 0 and current_mp_features[i+1] == 0 and current_mp_features[i+2] == 0:
                        flag = False
                        break
            # if this date survives, create a a data line for this date and for this ID
            if flag:
                data_line_mp = []
                for task in dictionary['metadata']['tasks_required']:
                    data_line_mp.extend(dictionary['data'][ID][date][task + '_mp_features'])
                data_line = [ID,date] + data_line_mp
                # print(data_line)
                features_df = pd.concat([features_df, pd.DataFrame([data_line], columns=columns)])

    return features_df

# function to read a single inout csv file (that contains the openface action unit data), and generate the action unit features
def extract_au_features(openface_file, task, dictionary):
    try:
        raw_openface_features = pd.read_csv(openface_file)
        data_line = []
        for AU_value in dictionary['metadata']['AU_values'][task]:
            raw_openface_features['AU%02d' % AU_value] = raw_openface_features[' AU%02d_r' % AU_value].astype(float)
            mean = raw_openface_features['AU%02d' % AU_value].mean()
            variance = raw_openface_features['AU%02d' % AU_value].var()
            entropy = raw_openface_features['AU%02d' % AU_value].apply(lambda x: -x*np.log2(x) if x != 0 else 0).sum()
            entropy_from_scipy = ent(raw_openface_features['AU%02d' % AU_value])
            data_line.append(mean)
            data_line.append(variance)
            data_line.append(entropy_from_scipy)
    except Exception as e:
        print('error' + str(e) + ' with file: ' + openface_file)
        data_line = [0] * 21
    return data_line

# reads a directory having csv file with openface action unit extracts and generate the action unit feature dataset
def create_openface_features_dataframe(openface_extract_directory):
    dictionary = generate_metadata_and_data_dictionary(openface_extract_directory)
        
    for ID in tqdm(dictionary['data'], total=len(dictionary['data']), file=sys.stdout):
        for task in dictionary['metadata']['tasks_required']:
            for date in dictionary['data'][ID]:
                file_name = dictionary['data'][ID][date][task + '_file']
                au_features = extract_au_features(os.path.join(file_name), task, dictionary)
                dictionary['data'][ID][date][task + '_au_features'] = au_features    
    
        # we now have the mediapipe features for each of the videos
    # create a dataframe with columns as the userID, and mediapipe features
    columns = ['ID', 'date']

    for task in dictionary['metadata']['tasks_required']:
        for AU_value in dictionary['metadata']['AU_values'][task]:
            columns.append(task + '_AU%02d' % AU_value + '_mean')
            columns.append(task + '_AU%02d' % AU_value + '_var')
            columns.append(task + '_AU%02d' % AU_value + '_entropy')
            
    features_df = pd.DataFrame(columns=columns)

    # we now have the mediapipe features for each of the videos
    # for a particular user, if there are multiple videos and any of those vidoes videos fails to generate the mediapipe features, then we will not consider the mediapipe features for that user
    # in that case discard the mediapipe features for that user and discard the video file name from the user's data
    for ID in dictionary['data']:
        for date in dictionary['data'][ID]:
            flag = True
            for task in dictionary['metadata']['tasks_required']:
                data_line_au = []
                for task in dictionary['metadata']['tasks_required']:
                    data_line_au.extend(dictionary['data'][ID][date][task + '_au_features'])
                data_line = [ID,date] + data_line_au
                # print(data_line)
            features_df = pd.concat([features_df, pd.DataFrame([data_line], columns=columns)])

    return features_df


# function to combine the action unit and landmark feature dataset
def combine_openface_mediapipe_features(openface_features_df, mediapipe_features_df):
    """
    Combine openface and mediapipe features into one dataframe based on the ID and date
    """
    combined_openface_mediapipe_features_df = pd.merge(openface_features_df, mediapipe_features_df, on=['ID'])
    combined_openface_mediapipe_features_df.to_csv('combined_openface_mediapipe_features.csv', index=False)
    return combined_openface_mediapipe_features_df
    
        



def main(args):
    print('\n\nCreating mediapipe features dataframe\n\n')
    mediapipe_features_df = create_mediapipe_features_dataframe(args.video_files_directory)
    mediapipe_features_df.to_csv(args.output_dir + 'mediapipe_features.csv', index=False)
    #mediapipe_features_df = pd.read_csv(args.output_dir + 'mediapipe_features.csv')
    mediapipe_features_df['ID'] = mediapipe_features_df['ID'].astype(str).str[:-4]
    print('\n\nCreating openface features dataframe\n\n')   
    openface_features_df = create_openface_features_dataframe(args.openface_files_directory)
    openface_features_df.to_csv(args.output_dir + 'openface_features.csv', index=False)
    #openface_features_df = pd.read_csv(args.output_dir + 'openface_features.csv')
    openface_features_df['ID'] = openface_features_df['ID'].astype(str).str[:-4]
    print('\n\nCombining mediapipe and openface features\n\n')
    combined_openface_mediapipe_features_df = combine_openface_mediapipe_features(openface_features_df, mediapipe_features_df)
    combined_openface_mediapipe_features_df.to_csv(args.output_dir + 'combined_openface_mediapipe_features.csv', index=False)
    print('\n\nFeature file generated and saved\n\n')


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_files_directory', type=str, default='raw_videos')
    parser.add_argument("--openface_files_directory", type=str, default='openface_extracts')
    parser.add_argument("--output_dir", type=str, default='may_22/')
    args = parser.parse_args()

    main(args) 
    



