import argparse
import deep_sheep as ds 
from pathlib import Path
import pandas as pd

parser = argparse.ArgumentParser(description='Deep Sheep Classification Settings')

parser.add_argument('model_path', type=str, help='Model File')
parser.add_argument('sort_dir', type=str, help='Directory to run classification on')
parser.add_argument('output_file', type=str, help='File for exporting results')
parser.add_argument('-MC_runs', type=int, default=10, help='Number of Monte Carlo runs for each image for uncertainty quantification')
parser.add_argument('-saved_progress_file', default=None, type=str, help='File of saved progress due to previous run / early terminated job')
parser.add_argument('-group_segmentation', type=bool, default=True, help='Toggles running group segmentation - default is True')
parser.add_argument('-sort_files', type=bool, default=False, help='Toggles reorganizing files - default is False')
parser.add_argument('-probability_threshold', type=float, default=0.5, help='Adjusts probability threshold for determining sheep')
parser.add_argument('-group_interval', type=int, default=30, help='Minutes for determining time group segmentation - i.e., if images occur within 30 min of each other, they are in the same group')

args = parser.parse_args()

# Parse arguments for settings 
target_directory = args.sort_dir
model_path = args.model_path
MC_runs = args.MC_runs
output_filename = args.output_file
progress_file = args.saved_progress_file
group_segmentation = args.group_segmentation
sort_files = args.sort_files 
probability_threshold = args.probability_threshold 
group_interval = args.group_interval

if progress_file is None: 
	print('No progress file given, reading directory...')
	img_df = ds.get_images_in_dir(target_directory)

else:
	print('Using progress file %s'%(progress_file))
	img_df = pd.read_csv(progress_file)


ds.export_img_df(img_df, output_filename)

# Run predictions on target dir 
img_df = ds.classify_dir(target_directory, img_df, str(model_path), MC_runs,
						output_filename, progress_file, probability_threshold)

# Export results
ds.export_img_df(img_df, output_filename)

if group_segmentation:
    img_df = ds.group_segmentation(img_df, group_interval)

#if sort_files:
#    ds.sort_files(img_df) 

# Export results
ds.export_img_df(img_df, output_filename)




