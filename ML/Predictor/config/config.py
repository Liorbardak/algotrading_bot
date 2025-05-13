import json
import os
def get_config():
    '''
    Read the config file
    '''
    params = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),  'training_config.json')))
    params['features'] = params['features'].split(',')
    params['input_len'] = len(params['features'])
    return params
