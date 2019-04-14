from webapp import app
from flask import render_template, request
from os import listdir

import json
import numpy as np
import pickle

from webapp.models import dwb_model

@app.route('/')
@app.route('/index')
def index():
  return render_template('index.html', title='Home')

# Example of how to add backend model to frontend template.
@app.route('/dwb')
def dwb():
  return render_template('deepwordbug.html',
      title='DeepWordBug Text Sequence',
      model_name="dwb",)
  
# Actually run the model
@app.route('/run_adversary', methods=['POST'])
def run_adversary():
  print('Starting adversary generation')
  model_name    = request.form['model_name']

  if model_name == 'dwb':
    # Get input string + other parameters
    s = request.form['input_string']
    model_num = request.form['dwb_model_num']
    power = int(request.form['dwb_power'])
    scoring = request.form['dwb_scoring']
    transform = request.form['dwb_transform']

    original_class, adversary_class, adv_example, orig_likelihoods, adv_likelihoods, classes_list, max_scores, = dwb_model.visualize(
      s, model_num, power, scoring, transform)

    print(s)
    print(adversary_class)
    print(adv_example)
    print(orig_likelihoods)
    print(adv_likelihoods)
    print(classes_list)

  # else if: add your code here!

  ret_val = {
    'original_class': original_class, 
    'adversary_class': adversary_class,
    'original_text_data': [s],
    'adv_text_data': [adv_example],

    'orig_likelihood_data': [{
      'x': classes_list,
      'y': [float(y) for y in orig_likelihoods],
      'marker': dict(color='rgb(26, 118, 255)'),
      'type':'bar'
    }],
    'adv_likelihood_data': [{
      'x': classes_list,
      'y': [float(y) for y in adv_likelihoods],
      'marker': dict(color='rgb(26, 118, 255)'),
      'type':'bar'
    }],

    'max_scores': max_scores.tolist(),
  }

  return json.dumps(ret_val)
