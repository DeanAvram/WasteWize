#! usr!bin

from flask import Flask, jsonify, abort, make_response, url_for
from flask import request
from socket import *
import numpy as np
import Network
import torch
# from flask_httpauth import HTTPBasicAuth

sock = socket(AF_INET, SOCK_STREAM)
sock.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
sock.bind(('0.0.0.0', 0))

app = Flask(__name__)
# auth = HTTPBasicAuth()


invalid_Task_JSON_404 = {'error': 'Invalid Task Id'}
invalid_Task_JSON_400 = {'error': 'Bad Request'}

'''
@auth.get_password
def get_password(username):
    if username == 'dean':
        return 'avram'

    return None


@auth.error_handler
def unauthorized():
    return make_response(jsonify({'Response': 'Invalid credentials'}), 401)  # 403 to prevent dialog box
'''
device = Network.get_default_device()
model = Network.ResNet()
Network.to_device(model, device)

model.load_state_dict(torch.load("model_file.pt", map_location=torch.device('cpu')))
model.eval()


@app.route('/predict', methods=['POST'])
def get_prediction_post_request():
    # Works only for a single sample
    if request.method == 'POST':
        data = request.get_json()  # Get data posted as a json
        image_name = data['image']
        # data = np.array(data)[np.newaxis, :]  # converts shape from (4,) to (1, 4)
        # prediction = model.predict(data)  # runs globally loaded model on the data
        prediction = Network.predict_external_image(model, image_name)
    return prediction


@app.route('/predict/<image_name>', methods=['GET'])
def get_prediction_get_request(image_name):
    # Works only for a single sample
    if request.method == 'GET':
        # data = request.get_json()  # Get data posted as a json
        image_name = request.view_args['image_name']
        image_name = image_name.replace('.', '/') + '.jpg'
        # data = np.array(data)[np.newaxis, :]  # converts shape from (4,) to (1, 4)
        # prediction = model.predict(data)  # runs globally loaded model on the data
        prediction = Network.predict_external_image(model, image_name)
    return prediction


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify(invalid_Task_JSON_404), 404)


@app.errorhandler(400)
def not_found(error):
    return make_response(jsonify(invalid_Task_JSON_400), 400)


app.run('127.0.0.1', 5000)
