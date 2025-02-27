from flask import Flask, jsonify, request
from models.EnV2 import EnV2
from models.En import En
from models.Ar import Ar
from models.De import De
from models.Nl import Nl
from models.Pt import Pt
from sentence_transformers import SentenceTransformer
import os
import pickle
import torch 
import numpy as np 
import pandas as pd
from configs.modelsConfig import conf 
from flask import render_template




app = Flask(__name__)

loaded_model = ''
model = None
concepturis = []
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = 'cpu'
# return all models

@app.route('/', methods=['POST', 'GET'])
@app.route('/index', methods=['POST', 'GET'])
def index():
    print('salam')
    return render_template('index.html', title='Welcome')

@app.route("/all-models", methods=['POST', 'GET'])
def all_models():
    return jsonify(list(conf.keys()))


# get batch of occupations title and descriptions and return occupation conceptUries
@app.route("/occupations-detection", methods=['POST', 'GET'])
def occupation_detection():
    content = request.json
    items = content['items']
    texts = [get_text(item['title'], item['description']) for item in items]    
    check_and_change_if_needed_loaded_model(content['model'])
    
    if is_classifer(content['model']):
        probs = classifier_predict_similarity(texts)
    else:
        probs = cosine_embeds_similarity(texts)
    top_indices = get_best_indices(probs, content['topk'])
    return make_output(top_indices)



########################################

def get_text(title, descriptioin):
    return title+' '+descriptioin


def check_and_change_if_needed_loaded_model(model_name):
    global loaded_model, concepturis, DEVICE
    if loaded_model != model_name:
        base_path = "./models"
        drpath = os.path.join(base_path,"weights",conf[model_name]["lang"])
        model_path = os.path.join(drpath,"model.pt")
        if load_model(model_name, model_path,  DEVICE):
            loaded_model = model_name
            concepturis_path = os.path.join(drpath,'concepturis.pkl')
            with open(concepturis_path,'rb') as f:
                concepturis = pickle.load(f)
        
    return True


# initialization (load models by name)
def load_model(model_name, model_path, device):
    global model
    if model_name == "EnV2":
        model = EnV2()
        model.load_state_dict(torch.load(model_path,map_location=device))
    elif model_name == "En":
        model = En(device)
    elif model_name == "Ar":
        model = Ar(device)
    elif model_name == "De":
        model = De(device)
    elif model_name == "Nl":
        model = Nl(device)
    elif model_name == "Pt":
        model = Pt(device)

    return True
    


def is_classifer(model_name):
    if conf[model_name]["model_type"]=="classifier":
        return True
    return False


def get_best_indices(probablities, topk):
    return  torch.topk(probablities, topk,dim=-1).indices.cpu().data.numpy()


def make_output(top_indices):
    global concepturis
    return jsonify([[concepturis[i] for i in top_indices[i]] for i in range(len(top_indices))])


def classifier_predict_similarity(texts):
    global model
    train_encoded_input = model.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    return model(train_encoded_input)


def cosine_embeds_similarity(texts):
    global model
    return model.classify(texts)
