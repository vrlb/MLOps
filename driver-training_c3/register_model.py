# Import libraries
import argparse
import joblib
from azureml.core import Workspace, Model, Run

# Get parameters
parser = argparse.ArgumentParser()
parser.add_argument('--model_folder', type=str, dest='model_folder', default="driver_model", help='model location')
args = parser.parse_args()
model_folder = args.model_folder

# Get the experiment run context
run = Run.get_context()

# Load the model
print("Loading model from " + model_folder)
model_file = model_folder + "/driver_model.pkl"
model = joblib.load(model_file)

Model.register(workspace=run.experiment.workspace,
               model_path = model_file,
               model_name = 'driver_model',
               tags={'Training context':'Pipeline'})

run.complete()
