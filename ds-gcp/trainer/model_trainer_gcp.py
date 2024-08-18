from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input, Xception
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from pathlib import Path
import argparse
import subprocess


def get_dropout(input_tensor, p=0.25, mc=False):
    if mc:
        return Dropout(p)(input_tensor, training=True)
    else:
        return Dropout(p)(input_tensor)


def run_command(command: list):
	try:
	    result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	    print("Success:", result.stdout.decode('utf-8'))
	except subprocess.CalledProcessError as e:
	    print("Error:", e.stderr.decode('utf-8'))



parser = argparse.ArgumentParser()
parser.add_argument('--training_directory', 
					type=str, 
					required=True)
parser.add_argument('--train_from_base_model', 
					action='store_true')  
parser.add_argument('--base_trained_model', 
					type=str, 
					required=False)
parser.add_argument('--output_file', 
					type=str,
					required=True) 

args = parser.parse_args()

training_directory = args.training_directory
train_from_base_model = args.train_from_base_model
base_trained_model = args.base_trained_model
output_model = args.output_file


# Copy files from input bucket to local. 
copy_training_data_command = ['gsutil', '-m', 'cp', '-r', training_directory, '.']
run_command(copy_training_data_command)


# Prep training dataset 
training_directory = training_directory.split('/')[-1]

train_datagen=ImageDataGenerator(vertical_flip=True, rotation_range=45,
                                 brightness_range=(0.3, 0.9), preprocessing_function=preprocess_input)

train_generator=train_datagen.flow_from_directory(Path(training_directory),
                                                 target_size=(249,249),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='binary',
                                                 shuffle=True)

# Initialize model
# Due to issues with arguments across tensorflow versions, 
#   we build the base model architecture and then populate it with weights (if desired) instead of serializing. 
base_model = Xception(weights='imagenet',include_top=False)
x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x) 
x=Dense(1024,activation='relu')(x) 
x = get_dropout(x, p=0.25, mc=True)
x=Dense(1024,activation='relu')(x) 
x = get_dropout(x, p=0.25, mc=True)
x=Dense(512,activation='relu')(x) 
preds=Dense(1,activation='sigmoid')(x)
model=Model(inputs=base_model.input,outputs=preds)

if train_from_base_model: 
	
	# Fetch from cloud bucket
	copy_model_command = ['gsutil', '-m', 'cp', '-r', base_trained_model, '.']
	run_command(copy_model_command)
	
	# Load weights 
	base_trained_model = base_trained_model.split('/')[-1]
	print(f'Loading base_trained_model, {base_trained_model}')
	model.load_weights(Path(base_trained_model))

# Make last layers of model trainable
for layer in model.layers[:-20]:
    layer.trainable=False
for layer in model.layers[-20:]:
    layer.trainable=True

# Compile model
model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])

step_size_train=train_generator.n//train_generator.batch_size

# Train model 
model.fit(train_generator,
       steps_per_epoch=step_size_train,
       verbose=1,
       use_multiprocessing=True,
       epochs=10,
       # trying to place greater weight on sheep images in running cost function
       class_weight={0:1, 1:3})


# Save model and push back to GCS
model.save(Path(output_model))

copy_saved_model_command = ['gsutil', '-m', 'cp', '-r', output_model, 'gs://deep-sheep-trained-models/']
run_command(copy_saved_model_command)