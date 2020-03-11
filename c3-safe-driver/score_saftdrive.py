import json
import joblib
import numpy as np
from azureml.core.model import Model

# Called when the service is loaded
def init():
    global model
    # Get the path to the deployed model file and load it
    model_path = Model.get_model_path('driver_model')
    model = joblib.load(model_path)

# Called when a request is received
def run(raw_data, request_headers):
    # Get the input data as a numpy array
    data = np.array(json.loads(raw_data)['data'])
    # Get a prediction from the model
    predictions = model.predict(data)
    # Get the corresponding classname for each prediction (0 or 1)
    
    
    print(('{{"RequestId":"{0}", '
       '"TraceParent":"{1}", '
       '"NumberOfPredictions":{2}}}'
       ).format(
           request_headers.get("X-Ms-Request-Id", ""),
           request_headers.get("Traceparent", ""),
           len(predictions)
    ))
    
    
    classnames = ['safe-driver', 'not-safe-driver']
    predicted_classes = []
    for prediction in predictions:
        predicted_classes.append(classnames[int(prediction)])
    # Return the predictions as JSON
    return json.dumps(predicted_classes)
