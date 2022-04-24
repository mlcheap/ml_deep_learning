from flask import Flask

app = Flask(__name__)

# initialization (load models by name)
def load_models(model_name):
    pass

# return all models
@app.route("/all-models", methods=['POST', 'GET'])
def all_models():
    return "en"

# get batch of occupations title and descriptions and return occupation conceptUries
@app.route("/occupation-detection/batch")
def occupation_detection_batch():
    return "<p>Hello, World!</p>"

# get single occupation title and description and return occupation conceptUri
@app.route("/occupation-detection/single")
def occupation_detection_single():
    return "<p>Hello, World!</p>"
