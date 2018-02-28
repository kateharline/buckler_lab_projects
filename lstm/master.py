
# master function to parallelize tuning on the cbsu

import os
import platform
import itertools as it

import datain as d
import modelmaker as m
from models import cnn, lstm

def tuple_combos(dict):
    keys = sorted(dict)
    combinations = it.product(*(dict[key] for key in keys))

    return combinations


def run_a_model(model_type, classify, model_dir, t_params, data):
    '''
    run an instance of the given model
    :param data: tuple of train, test and val x+y values
    :param model_type: module of model to run
    :param t_params: tuple of values for lr, num of layers and units per layer
    '''
    lr = t_params[0]
    num_layers = t_params[1]
    units_per_layer = t_params[2]

    model_name = str(model_type)+'_lr_'+str(lr)+'_lys_'+str(num_layers)+'_uns_'+str(units_per_layer)

    m.main(model_type, classify, model_name, model_dir, t_params, *data)

    return None

def iterate_params(model_type, classify, model_dir, tuning_params, data):
    '''
    iteratively evaluate models with different combinations of parameters
    :param data: tuple of train, test and val x+y values
    :param model_type: module of model to run
    :param tuning_params: dictionary of tuning parameters assoc with lists of values to iteratively try
    '''
    # make array of tuples all combinations of tuning params
    combinations = tuple_combos(tuning_params)

    for combo in combinations:
        run_a_model(model_type, classify, model_dir, combo, data)

    return None

def iterate_models(model_dir, models, tuning_params):
    model_combos = tuple_combos(models)

    for combo in model_combos:
        model_type = combo[0]
        classify = combo[1]
        data_type = combo[2]

        # trying to manage gpu loads
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if model_type == cnn:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "1"


        data = d.main(data_type=data_type, categorical=classify, standardized=(not classify))
        iterate_params(model_type, classify, model_dir, tuning_params, data)

    return None


def main():
    if 'Ubuntu' in platform.platform():
        os.chdir('/home/kh694/Desktop/buckler-lab/box-data')
    elif 'Darwin' in platform.platform():
        os.chdir('/Users/kateharline/Desktop/buckler-lab/box-data')
    else:
        ## need to fit to cbsu/file structure
        os.chdir('/home/wkdir/lstmOrCnn/data')


    output_folder = 'outputs'
    os.system('mkdir ' + output_folder)

    model_dir = os.path.join(output_folder, 'tmp')
    os.system('mkdir ' + model_dir)

    tuning_params = { 'lr' : [.001, .0001, .00001],
                      'num_layers' : [1, 2, 3, 4],
                      'units' : [32, 64, 128],
                      }

    models = {
        'model_type' : [cnn, lstm],
        'classify' : [True, False],
        'data_type' : ['random', 'balanced', 'unbalanced']
    }

    iterate_models(model_dir, models, tuning_params)





if __name__ == '__main__':
    main()