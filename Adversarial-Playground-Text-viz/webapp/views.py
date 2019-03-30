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

@app.route('/dwb')
def dwb():
  return render_template('deepwordbug.html',
      title='DeepWordBug Text Sequence',
      model_name="dwb",)
  
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

    original_class, adversary_class, adv_example = dwb_model.visualize(s, model_num, power, scoring, transform)

    print(s)
    print(adversary_class)
    print(adv_example)

    ret_val = {
      'original_class': original_class, 
      'adversary_class': adversary_class,
      'original_text_data': [s],
      'adv_text_data': [adv_example],
    }
    return json.dumps(ret_val)

  print(model_name)

  # if model_name == 'fjsma':
  #   # Perform tensor flow request
  #   upsilon_value = request.form['attack_param']
  #   target_value  = int(request.form['target'])
  #   print('Performing the fjsma L0 attack from {} to {}'.format(seed_class, target_value))

  #   adversary_class, adv_example, adv_likelihoods = l0_model.attack(seed_image, target_value, upsilon_value, fast=True)

  # elif model_name =='Linf':
  #   epsilon_value = request.form['attack_param']
  #   adversary_class, adv_example, adv_likelihoods = linf_model.fgsm(seed_image, seed_class, epsilon_value)
    
  # print('New adversary is classified as {}'.format(adversary_class))
  # ret_val = {
  #             'adversary_class':str(adversary_class), 
  #             'image_data': [{
  #                 'z' : list(reversed(adv_example.tolist())) if adv_example is not None else '',
  #                 'type': 'heatmap',
  #                 'colorscale': [
  #                     ['0.0',            'rgb(0.00,0.00,0.00)'],
  #                     ['0.111111111111', 'rgb(28.44,28.44,28.44'],
  #                     ['0.222222222222', 'rgb(56.89,56.89,56.89)'],
  #                     ['0.333333333333', 'rgb(85.33,85.33,85.33)'],
  #                     ['0.444444444444', 'rgb(113.78,113.78,113.78)'],
  #                     ['0.555555555556', 'rgb(142.22,142.22,142.22)'],
  #                     ['0.666666666667', 'rgb(170.67,170.67,170.67)'],
  #                     ['0.777777777778', 'rgb(199.11,199.11,199.11)'],
  #                     ['0.888888888889', 'rgb(227.56,227.56,227.56)'],
  #                     ['1.0',            'rgb(256.00,256.00,256.00)']
  #                   ],
  #                 'showscale':'false',
  #                 'showlegend':'false',
  #               }],
  #               'likelihood_data': [{
  #                 'x':list(range(10)),
  #                 'y':[float(x) for x in adv_likelihoods],
  #                 'type':'bar'
  #               }],
  #           }
  # return json.dumps(ret_val)
  
