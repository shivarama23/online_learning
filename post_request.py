# -*- coding: utf-8 -*-
"""
Created on Sun May  8 21:27:28 2022

@author: shivaramakrishna.kv
"""

#make a POST request
import requests
sample_test = {"model_name": "new_model", "model_id": 12345}
# sample_test = {"sample": "Ceiling Fan"}
# res = requests.post('http://192.168.0.104:9000/training', json=sample_test)
res = requests.post('http://192.168.0.104:9000/prediction', json=sample_test)
print('response from server:', res.text)
# dictFromServer = res.json()