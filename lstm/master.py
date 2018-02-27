
# master function to parallelize tuning on the cbsu

import os
import platform
import itertools as it

import datain as d


def run_a_model(data, model_dir, model_type, t_params):
    '''
    run an instance of the given model
    :param data: tuple of train, test and val x+y values
    :param model_type: module of model to run
    :param t_params: tuple of values for lr, num of layers and units per layer
    '''
    model_name = str(model_type)+'_lr_'+str(t_params[0])+'_lys_'+str(t_params[1])+'_uns_'+str(t_params[2])

    model_type.main(model_name, model_dir, t_params, *data)

    return None

def iterate_models(data, model_dir, model_type, tuning_params):
    '''
    iteratively evaluate models with different combinations of parameters
    :param data: tuple of train, test and val x+y values
    :param model_type: module of model to run
    :param tuning_params: dictionary of tuning parameters assoc with lists of values to iteratively try
    '''
    # make array of tuples all combinations of tuning params
    params = sorted(tuning_params)
    combinations = it.product(*(tuning_params[param] for param in params))

    for combo in combinations:
        run_a_model(data, model_dir, model_type, combo)

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

    # categorical data sets
    # returns (train_encoded, train, test_encoded, test, val_encoded, val)
    random = d.main(data_type='random', categorical=True, standardized=False)
    iterate_models(random, model_dir, cnnClass, tuning_params)
    iterate_models(random, model_dir, lstmClass, tuning_params)

    bal = d.main(data_type='balanced', categorical=True, standardized=False)
    iterate_models(bal, model_dir, cnnClass, tuning_params)
    iterate_models(bal, model_dir, lstmClass, tuning_params)

    unbal = d.main(data_type='unbalanced', categorical=True, standardized=False)
    iterate_models(unbal, model_dir, cnnClass, tuning_params)
    iterate_models(unbal, model_dir, lstmClass, tuning_params)

    # continuous datasets
    random_cont = d.main(data_type='random', categorical=False, standardized=True)
    iterate_models(random_cont, model_dir, cnn, tuning_params)
    iterate_models(random_cont, model_dir, lstm, tuning_params)

    bal_cont = d.main(data_type='balanced', categorical=False, standardized=True)
    iterate_models(bal_cont, model_dir, cnn, tuning_params)
    iterate_models(bal_cont, model_dir, lstm, tuning_params)

    unbal_cont = d.main(data_type='unbalanced', categorical=False, standardized=True)
    iterate_models(unbal_cont, model_dir, cnn, tuning_params)
    iterate_models(unbal_cont, model_dir, lstm, tuning_params)




if __name__ == '__main__':
    main()