# Deep Sheep helper functions 

import os 
import sys
import numpy as np
from progressbar import ProgressBar
from pathlib import Path
from scipy.stats import iqr
from keras.preprocessing import image
from keras.models import load_model
from keras.applications.xception import preprocess_input
import pandas as pd
import math
from PIL import Image, ExifTags, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import exifread

# Set random number seed
np.random.seed(0)

# Get the list of all JPG files in target directory tree at given path   
def get_images_in_dir(directory):
	
	print('\n \n Getting images and extracting metadata...')
	image_list = list()
	corrupted_files = list()


	for (dirpath, dirnames, filenames) in os.walk(directory):
		for file in filenames:
			if file.lower().endswith('.jpg'):
			
				# Store path 
				dirpath = Path(dirpath)
				filepath = dirpath / file
				
				# Confirm filesize > 0
				if os.path.getsize(filepath) > 0:
					try: 
						# Get timestamp of image from metadata
						f = open(filepath, 'rb')
						tags = exifread.process_file(f, details=False)
						ts = str(tags['Image DateTime'])
						
						image_list.append([dirpath, filepath, file, ts])
					
					except Exception: 
						# Should only fail in instances where file cannot be opened
						corrupted_files.append(filepath)

	image_df = pd.DataFrame(image_list, columns=['parent_dir', 'full_path', 'filename', 'timestamp'])

	image_df ['timestamp'] = pd.to_datetime(image_df['timestamp'], format='%Y:%m:%d %H:%M:%S')
	image_df = image_df.sort_values(by=['parent_dir', 'timestamp'])
	image_df = image_df.reset_index(drop=True)

	corrupted_files_df = pd.DataFrame(corrupted_files)
	corrupted_files_df.to_csv('corrupted_files.csv', index=False)

	return image_df


def load_image(img_path):
	
	review_image = image.load_img(img_path, target_size = (224, 224))
	review_image = image.img_to_array(review_image)
	review_image = np.expand_dims(review_image, axis = 0)
	review_image = preprocess_input(review_image)

	return review_image



# Make predictions for given image based number of MC runs requested or otherwise stop early if variance is low enough
def classify_image(loaded_image, model, MC_runs, early_MC_stopping_enabled=True):

	# Run MC predictions
	MC_predictions = []
	for i in range(MC_runs): 

		MC_predictions.append(model.predict(loaded_image)[0][0])

		# Stop MC iterations early if variance in first 3 predictions is low enough or converges early
		if early_MC_stopping_enabled and len(MC_predictions) >=  3 and iqr(MC_predictions) < 0.01:
			break

	# Save image stats
	sheep_probability = np.mean(MC_predictions)
	prediction_var = np.var(MC_predictions)
	prediction_iqr = iqr(MC_predictions)

	return sheep_probability, prediction_var, prediction_iqr



# Flow for classifying all images in target directory and exporting results
# First pass prior to segmentation and sorting files
def classify_dir(target_directory, target_dir_img_df, model_path, MC_runs, output_file, progress_file, probability_threshold):
	
	if len(target_dir_img_df.columns) == 4:
		# Need placeholders for model results
		target_dir_img_df['sheep_probability'] = None
		target_dir_img_df['pred_var'] = None
		target_dir_img_df['pred_iqr'] = None

	# Load model
	print('\n \n Loading model... \n')
	model = load_model(model_path)

	# Run predictions on target directory images
	print('\n \n Running predictions on %s using %s' %(target_directory, model_path))

	pbar = ProgressBar()

	for img_path in pbar(target_dir_img_df[target_dir_img_df['sheep_probability'].isnull()]['full_path']):

			# Additional filter to catch files of size > 0 that are unable to be loaded 

			#load_success = False

			try: 
				# Load and preprocess image for model
				loaded_image = load_image(img_path)
				#load_success = True 

				# Get predictions for image
				p_sheep, pred_var, pred_iqr = classify_image(loaded_image, model, MC_runs)

				# Save image data
				index = target_dir_img_df[target_dir_img_df['full_path'] == img_path].index
				target_dir_img_df.loc[index, 'sheep_probability'] = p_sheep
				target_dir_img_df.loc[index, 'pred_var'] = pred_var
				target_dir_img_df.loc[index, 'pred_iqr'] = pred_iqr

				# Save intermediary temp file every 100 files 
				if index % 100 == 0:
					target_dir_img_df.to_csv(output_file, index=False)


			except Exception: 
				
				# There are enough filters that by now this should only be corrupted files of size > 0	
				continue

	target_dir_img_df['prediction'] = target_dir_img_df['sheep_probability'].apply(lambda x: 1 if x >= probability_threshold else 0)

	return target_dir_img_df


# Helper function to split images into groups based on occurence within 30 min (or whatever user sets as group interval threshold)
def initial_time_group_segmentation(img_df, group_interval):

	# Check next image directory and timestamp difference
	img_df = img_df.join(img_df[['parent_dir', 'timestamp']].shift(-1), rsuffix='_lead')
	img_df['time_delta'] = np.abs(img_df['timestamp_lead'] - img_df['timestamp'])
	
	# If next image is in same parent directory and within 30 min, label it in same group 
	img_df['lead_is_group_member'] = img_df[['time_delta', 'parent_dir', 'parent_dir_lead']].apply(lambda x: True 
												if (x['time_delta'] <= pd.Timedelta(minutes=group_interval)) 
												and (x['parent_dir'] == x['parent_dir_lead'])
												else False, axis=1)

	img_df['time_group'] = None
	group = None

	# Identify first timestamp of each group to create group label
	for idx, lead_is_group in enumerate(img_df['lead_is_group_member']):
		if lead_is_group == True:
			
			# Last member of previous group should reset group = None so this will set group name based on first in new group
			if group == None:       
				new_group_timestamp = img_df['timestamp'].iloc[idx]
				group = new_group_timestamp.strftime('%Y%m%d_%H%M')
			  
			# Assign initial group label (will add final group member timestamp later)
			img_df.loc[idx, 'time_group'] = group
		
		else:
			
			# If lead row is not in group, assign current group label and reset group name
			img_df.loc[idx, 'time_group'] = group
			group = None

	

	# Identify last timestamp of each group to finish group label (if it is different from first timestamp)
	for idx, group in enumerate(img_df['time_group']):
		

		if group != None:
			
			group_df = img_df[img_df['time_group'] == group]
			
			if len(group_df) > 0:
				end_time = group_df['timestamp'].max()
				end_time_str = end_time.strftime('%Y%m%d_%H%M')
				
				if end_time_str != group:
				
					final_group_name =  group + '-' + end_time_str
				
				else:
					
					final_group_name = group
					#rint(final_group_name)

				img_df.loc[idx, 'time_group'] = final_group_name 
		
		

	# If image is alone, then just assign group based on timestamp 

	img_df.loc[img_df[img_df['time_group'].isnull()].index, 'time_group'] = img_df[img_df['time_group'].isnull()]['timestamp'].apply(lambda x: x.strftime('%Y%m%d_%H%M'))




	return img_df[['parent_dir', 'full_path', 'filename', 'timestamp','time_group', 'sheep_probability', 'pred_var', 'pred_iqr', 'prediction']]


# For long continous groups of images where initial time segmentation fails, this will separate groups of sheep 
def long_group_segmentation(df):

	# DF is all images in folder and will be record for time groups 
	df = df.sort_values(by='timestamp').reset_index(drop=True)
	df['time_group'] = None
	
	# processing_df is copy that will be recursively updated to remove sorted images
	processing_df = df.copy()

	while len(processing_df) > 0:

		processing_df = processing_df[processing_df['time_group'].isnull()]

		# First check if there are any sheep 
		if len(processing_df[processing_df['prediction'] == 1]) > 0:

			first_sheep_timestamp = processing_df[processing_df['prediction'] == 1]['timestamp'].min()
			end_sheep_group = False

			# Identify point 30 minutes before first sheep  
			buffered_sheep_group_start = first_sheep_timestamp - pd.Timedelta(minutes=30)

			# Subset all images prior to buffered group start time
			pre_sheep_group = processing_df[processing_df['timestamp'] < buffered_sheep_group_start]

			if len(pre_sheep_group) > 0:

				# Set group for (non-sheep) images prior to buffered sheep start time 
				df.loc[pre_sheep_group.index[0]:pre_sheep_group.index[-1], 'time_group'] = (
					pre_sheep_group['timestamp'].min().strftime('%Y%m%d_%H%M') + '_' + buffered_sheep_group_start.strftime('%Y%m%d_%H%M'))

				# Remove segmented images from processing df 
				processing_df = df[df['time_group'].isnull()]

			# Want to find last sheep image and create segment ending 10 min after 
			check_buffer_min = first_sheep_timestamp

			# Recursively look forward in time until there are no sheep images, then pull the last sheep image and set end of sheep group 30 min after that point
			while not end_sheep_group:

				check_buffer_max = check_buffer_min + pd.Timedelta(minutes=30)
				check_buffer_df = processing_df[(processing_df['timestamp'] > check_buffer_min) & (processing_df['timestamp'] < check_buffer_max)]    

				if check_buffer_df['prediction'].sum() > 0:
					check_buffer_min = check_buffer_df[check_buffer_df['prediction'] == 1]['timestamp'].max()

				else: 
					buffered_sheep_group_end = check_buffer_max
					end_sheep_group = True 

			sheep_group = processing_df[(processing_df['timestamp'] >= buffered_sheep_group_start) & 
										(processing_df['timestamp'] <= buffered_sheep_group_end)]

			# Label tracking df with buffered sheep group label 
			if len(sheep_group) > 0: 
				df.loc[sheep_group.index[0]:sheep_group.index[-1], 'time_group'] = (
						buffered_sheep_group_start.strftime('%Y%m%d_%H%M') + '_' + buffered_sheep_group_end.strftime('%Y%m%d_%H%M'))

				processing_df = df[df['time_group'].isnull()]

		else:

			# If there are no sheep, then just label as one time group 
			df.loc[processing_df.index, 'time_group'] = processing_df['timestamp'].min().strftime('%Y%m%d_%H%M') + '_' + processing_df['timestamp'].max().strftime('%Y%m%d_%H%M')
			processing_df = df[df['time_group'].isnull()]

	return df 



def group_segmentation(images_df, group_interval):
	print('\n \n Running group segmentation...')


	images_df['timestamp'] = pd.to_datetime(images_df['timestamp'])

	# Per Ashley - usual cutoff is 30 minutes without sheep triggers the cutoff of a group. 
	# Run initial segmentation separating images > 30 min apart
	initial_segmentation =  initial_time_group_segmentation(images_df, group_interval)

	# Get min/max timestamps and number of images in each group 
	group_summary = initial_segmentation.groupby('time_group').agg({'timestamp':['min', 'max'], 'filename':'size'})
	group_summary.columns =  group_summary.columns.droplevel(0)
	group_summary = group_summary.reset_index()
	group_summary.columns = ['time_group', 'min_timestamp', 'max_timestamp', 'num_images']	

	group_summary['group_duration'] = group_summary['max_timestamp'] - group_summary['min_timestamp']

	# If duration >= 2 hours flag as long group and run long group segmentation 	
	group_summary['long_group'] = group_summary['group_duration'].apply(lambda x: 1 if x >= pd.Timedelta(hours=2) else 0)

	segmented_df = initial_segmentation[initial_segmentation['time_group'].isin(group_summary[group_summary['long_group'] == 0]['time_group'])]

	# For longer time groups, segment based on 30 min buffers surrounding sheep appearances 
	for group in group_summary[group_summary['long_group'] == 1]['time_group']:
		long_group_df = initial_segmentation[initial_segmentation['time_group'] == group]
		segmented_df = segmented_df.append(long_group_segmentation(long_group_df), ignore_index=True)

	return segmented_df




def sort_files(images_df):
	print('\n \n Sorting files')
	
	


def export_img_df(images_df, output_filename):
	# Confirm file name is valid
	if not str(output_filename).endswith('.csv'):
		output_filename = output_filename + '.csv'

	# Export dataframe
	print('\n \n Exporting to %s'%(output_filename))
	images_df.to_csv(output_filename, index=False)




