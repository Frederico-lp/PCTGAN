import os
import torch
import numpy as np
import pandas as pd


def save_models(model, model_name, dataset_name):
    if not os.path.exists('./saved_models'):
        os.makedirs('./saved_models')

    torch.save(model, './saved_models/' + model_name + '_' + dataset_name + '.pt')

def load_models(model_name, dataset_name):
    model = torch.load('./saved_models/' + model_name + '_' + dataset_name + '.pt')
    return model