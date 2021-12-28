import h5py
import numpy as np
np.random.seed(0)
import argparse
import math
from keras.layers import Dense,GlobalAveragePooling2D, Dropout
from keras.applications import MobileNet, xception
from keras.preprocessing import image
from keras.applications.xception import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam 
from keras.models import load_model
from sklearn.metrics import accuracy_score, roc_curve, auc, classification_report
import pandas as pd
import json

parser = argparse.ArgumentParser(description='Deep Sheep Model Training Settings')

parser.add_argument('training_directory', type=str, help='Directory to train model (with folders for classes)')
parser.add_argument('test_directory', type=str, help='Directory to test model (with folders for classes)')
parser.add_argument('saved_model_name', type=str, help='Name to use for saved model once trained')
parser.add_argument('-model_path', type=str, help='Existing model file')
parser.add_argument('-from_scratch', type=bool, default=False, help='Toggles whether to train model from scratch or using existing model file')
parser.add_argument('-epochs', type=bool, default=7, help='Number of training epochs')
parser.add_argument('-sheep_weight', type=float, default=3, help='Weight for sheep in loss function')
parser.add_argument('-sheep_threshold', type=float, default=0.5, help='Threshold for classifying sheep vs. non-sheep')

args = parser.parse_args()


# Parse arguments for settings 
training_directory = args.training_directory
test_directory = args.test_directory
saved_model_name = args.saved_model_name
model_path = args.model_path
from_scratch = args.from_scratch
epochs = args.epochs
sheep_weight = args.sheep_weight
sheep_threshold = args.sheep_threshold

# Specify class weights for loss function
class_weights = {0:1, 1:sheep_weight}

# Load training and test data
train_datagen=ImageDataGenerator(vertical_flip=True, rotation_range=45, 
                                 brightness_range=(0.3, 0.9), preprocessing_function=preprocess_input) 

train_generator=train_datagen.flow_from_directory(training_directory, # this is where you specify the path to the main data folder
                                                 target_size=(249,249),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='binary',
                                                 shuffle=True)


test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_generator = test_datagen.flow_from_directory(test_directory,
                                              target_size=(249,249),
                                              batch_size=32,
                                              color_mode='rgb',
                                              class_mode='binary',  
                                              shuffle=False)  # keep data in same order as labels

# Helper function for Monte Carlo dropout
def get_dropout(input_tensor, p=0.25, mc=False):
    if mc:
        return Dropout(p)(input_tensor, training=True)
    else:
        return Dropout(p)(input_tensor)

if from_scratch:
	# Load pretrained Xception model and create trainable layers 
	base_model = xception.Xception(weights='imagenet',include_top=False)
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

else:
	model = load_model(model_path)

for layer in model.layers[:-20]:
    layer.trainable=False
for layer in model.layers[-20:]:
    layer.trainable=True

# Compile and fit model 
model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])

step_size_train=train_generator.n//train_generator.batch_size

print('\n Fitting model on images in %s' %training_directory)

model.fit(train_generator,
                   steps_per_epoch=step_size_train,
                   validation_data = test_generator,
                   validation_steps = test_generator.n//test_generator.batch_size,
                   verbose=1,
                   use_multiprocessing=False,
                    epochs=epochs, 
                    # Adjust to change weight on sheep images in running cost function
                    class_weight=class_weights)


print('\n Evaluating model performance on images in %s' %test_directory)

# Prepare detailed performance summary
probabilities = model.predict(test_generator, steps = math.ceil(len(test_generator.filenames) / 32), verbose=1)
df = pd.DataFrame(zip(test_generator.filenames, [i[0] for i in probabilities]), columns = ['file', 'sheep_probability'])

df['prediction'] = df['sheep_probability'].apply(lambda x: 1 if x > sheep_threshold else 0)

true_label  = test_generator.labels
predictions = df['prediction']
probabilities = df['sheep_probability']

print ('Accuracy: %f' %(accuracy_score(true_label, predictions)))

fpr, tpr, _ = roc_curve(true_label, probabilities)
roc_auc = auc(fpr, tpr)
print ('ROC AUC: %0.2f' % roc_auc)

print ('\n Confusion Matrix')
print (pd.crosstab(true_label, np.array(predictions), rownames=['True'], colnames=['Predictions'], margins=True))

print ('\n Classification Report')
print (classification_report(true_label, predictions))



# Save model weights 
if not saved_model_name.endswith('.h5'):
	saved_model_name = str(saved_model_name) + '.h5'

model.save(saved_model_name) 
