import argparse
import deep_sheep as ds 
from pathlib import Path
import pandas as pd

parser = argparse.ArgumentParser(description='Deep Sheep Model Execution/Sorting Settings')

parser.add_argument('model_path', type=str, help='Model File')
parser.add_argument('sort_dir', type=str, help='Directory to run classification on')
parser.add_argument('output_file', type=str, help='File for exporting results')
parser.add_argument('-MC_runs', type=int, default=10, help='Number of Monte Carlo runs for each image for uncertainty quantification')
parser.add_argument('-saved_progress_file', default=None, type=str, help='File of saved progress due to previous run / early terminated job')
parser.add_argument('-group_segmentation', type=bool, default=True, help='Toggles running group segmentation - default is True. Turning this off will break sorting.')
parser.add_argument('-sort_files', type=bool, default=False, help='Toggles reorganizing files - default is False')
parser.add_argument('-destination_dir', type=str, default=None, help='Location to move time sorted groups')
parser.add_argument('-sort_in_place', type=bool, default=False, help='Whether to sort files in place or copy to destination directory.')
parser.add_argument('-sheep_probability_threshold', type=float, default=0.5, help='Adjusts probability threshold for determining sheep without review')
parser.add_argument('-non_sheep_probability_threshold', type=float, default=0.5, help='Adjusts probability threshold for determining non sheep without review')
parser.add_argument('-variance_threshold', type=float, default=0.002, help='Adjust variance threshold for determining whether or not review is needed')
parser.add_argument('-group_interval', type=int, default=30, help='Minutes for determining time group segmentation - i.e., if images occur within 30 min of each other, they are in the same group')
parser.add_argument('-long_group_threshold', type=int, default=120, help='Minutes for determining long time group segmentation - i.e., if group duration is longer than 120 min, run an extra segmentation pass through the group to find smaller groups of sheep')
parser.add_argument('-sheep_group_threshold', type=float, default=0.1, help='For each time group, proportion that must be sheep in ordered to be classified sheep without further review. Default is 0.1, 10%%.')
parser.add_argument('-need_review_group_threshold', type=float, default=0.4, help='For each time group, proportion that must be flagged for review in order to require group to be reviewed. Default is 0.4, 40%%.')

args = parser.parse_args()

# Parse arguments for settings 
target_directory = args.sort_dir
model_path = args.model_path
MC_runs = args.MC_runs
output_filename = args.output_file
progress_file = args.saved_progress_file
group_segmentation = args.group_segmentation
sort_files = args.sort_files 
destination_dir = args.destination_dir
sort_in_place = args.sort_in_place
sheep_probability_threshold = args.sheep_probability_threshold 
non_sheep_probability_threshold = args.non_sheep_probability_threshold 
variance_threshold = args.variance_threshold
group_interval = args.group_interval
long_group_threshold = args.long_group_threshold
sheep_group_threshold = args.sheep_group_threshold
need_review_group_threshold = args.need_review_group_threshold

if group_segmentation and not sort_in_place and destination_dir == None:
	print('Error - destination_dir must be specified if not sorting in place.')

elif group_segmentation and sort_in_place and destination_dir != None:
	print('Error - destination_dir specified when sorting in place.')

else:

	if not sort_files and destination_dir != None:
		print('\n Destination directory specified by sorting is not turned on. Will run model and export results without sorting.')

	if progress_file is None: 
		print('\n No progress file given, reading directory...')
		img_df = ds.get_images_in_dir(target_directory)

	else:
		print('Using progress file %s'%(progress_file))
		img_df = pd.read_csv(progress_file)

	if not output_filename.endswith('.csv'):
		output_filename = output_filename + '.csv'

	# Initial export of image dataframe
	ds.export_img_df(img_df, output_filename)

	# Run predictions on target dir 
	img_df = ds.classify_dir(target_directory, img_df, str(model_path), MC_runs,
							output_filename, progress_file, sheep_probability_threshold, non_sheep_probability_threshold, variance_threshold)

	#  Second export of image dataframe
	ds.export_img_df(img_df, output_filename)

	if group_segmentation:
		img_df, group_df = ds.group_segmentation(img_df, group_interval, long_group_threshold, sheep_group_threshold, need_review_group_threshold)
		group_df.to_csv('group-summary-' + output_filename, index=False)

	if sort_files:
		ds.sort_files(img_df, destination_dir, sort_in_place) 

	# Final export of image dataframe
	ds.export_img_df(img_df, output_filename)




