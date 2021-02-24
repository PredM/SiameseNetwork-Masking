import os
import sys
from datetime import datetime
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
# suppress debugging messages of TensorFlow
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from configuration.ConfigChecker import ConfigChecker
from configuration.Configuration import Configuration
from neural_network.Dataset import FullDataset
from neural_network.Optimizer import SNNOptimizer
from neural_network.SNN import initialise_snn
from neural_network.Inference import Inference
from baseline.Representations import Representation
from copy import copy
from configuration.Enums import BatchSubsetType


def change_model(config: Configuration, start_time_string, num_of_selction_iteration = None, get_model_by_loss_value = None):
    search_dir = config.models_folder
    loss_to_dir = {}

    for subdir, dirs, files in os.walk(search_dir):
        for directory in dirs:

            # Only "temporary models" are created by the optimizer, other shouldn't be considered
            if not directory.startswith('temp_snn_model'):
                continue

            # The model must have been created after the training has began, older ones are from other training runs
            date_string_model = '_'.join(directory.split('_')[3:5])
            date_model = datetime.strptime(date_string_model, "%m-%d_%H-%M-%S")
            date_start = datetime.strptime(start_time_string, "%m-%d_%H-%M-%S")

            if date_start > date_model:
                continue

            # Read the loss for the current model from the loss.txt file and add to dictionary
            path_loss_file = os.path.join(search_dir, directory, 'loss.txt')

            if os.path.isfile(path_loss_file):
                with open(path_loss_file) as f:

                    try:
                        loss = float(f.readline())
                    except ValueError:
                        print('Could not read loss from loss.txt for', directory)
                        continue

                    if loss not in loss_to_dir.keys():
                        loss_to_dir[loss] = directory
            else:
                print('Could not read loss from loss.txt for', directory)

    if num_of_selction_iteration == None and get_model_by_loss_value == None:
        # Select the best loss and change the config to the corresponding model
        min_loss = min(list(loss_to_dir.keys()))
        config.filename_model_to_use = loss_to_dir.get(min_loss)
        config.directory_model_to_use = config.models_folder + config.filename_model_to_use + '/'

        print('Model selected for inference:')
        print(config.directory_model_to_use, '\n')

    elif num_of_selction_iteration is not None and get_model_by_loss_value == None:
        # Select k-th (num_of_selction_iteration) best loss and change the config to the corresponding model
        loss_list = (list(loss_to_dir.keys()))
        loss_list.sort()
        min_loss = min(list(loss_to_dir.keys()))
        if num_of_selction_iteration < config.number_of_selection_tests_for_previous_models:
            selected_loss = loss_list[num_of_selction_iteration]
            print("Loss with index: ", num_of_selction_iteration, " selected.")
        else:
            idx = (num_of_selction_iteration - config.number_of_selection_tests_for_previous_models) * num_of_selction_iteration
            print("loss with index: ", idx," selected.")
            selected_loss = loss_list[idx]
        config.filename_model_to_use = loss_to_dir.get(selected_loss)
        config.directory_model_to_use = config.models_folder + config.filename_model_to_use + '/'

        print("Selection: ", num_of_selction_iteration, ' for model with loss: ', selected_loss, "(min loss:", min_loss,")", 'selected for evaluation on the validation set:')
        print(config.directory_model_to_use, '\n')
        return selected_loss

    elif get_model_by_loss_value is not None:
        # Select a model by a given loss value (as key) and change the config to the corresponding model
        config.filename_model_to_use = loss_to_dir.get(get_model_by_loss_value)
        config.directory_model_to_use = config.models_folder + config.filename_model_to_use + '/'

        print('Model selected for inference by a given key (loss):')
        print(config.directory_model_to_use, '\n')


# noinspection DuplicatedCode
def main():
    config = Configuration()

    # Define different version of original configuration
    #config_2 = copy(config)
    #config_2.hyper_file = config_2.hyper_file_folder + 'cnn2d_withAddInput_Graph_o1_GlobAtt_o2_2_HO_2_a.json'  # wie Standard, aber owl2vec als Graph Features added
    config_3 = copy(config)
    config_3.hyper_file = config_3.hyper_file_folder + 'cnn2d_withAddInput_Graph_o1_GlobAtt_o2_2_HO_2_b.json'  # wie Standard, aber Linear transformation an
    config_4 = copy(config)
    config_4.hyper_file = config_4.hyper_file_folder + 'cnn2d_withAddInput_Graph_o1_GlobAtt_o2_2_HO_2_c.json'  # wie Standard, aber nur Context Ausgabe

    ####
    '''
    config_2 = copy(config)
    config_2.batch_distribution = {
        BatchSubsetType.DISTRIB_BASED_ON_DATASET: 0.75,
        BatchSubsetType.EQUAL_CLASS_DISTRIB: 0.25
        }
    
    config_3 = copy(config)
    config_3.hyper_file = config_3.hyper_file_folder + 'cnn2d_withAddInput_Graph_o1_GlobAtt_o2_2_HO_2.json' # Owl2vec after 2DCNN Removed, film on
    config_4 = copy(config)
    config_4.hyper_file = config_4.hyper_file_folder + 'cnn2d_withAddInput_Graph_o1_GlobAtt_o2_2_HO_3.json'  # Owl2vec after 2DCNN Removed, film off
    config_5 = copy(config)
    config_5.hyper_file = config_5.hyper_file_folder + 'cnn2d_withAddInput_Graph_o1_GlobAtt_o2_2_HO_4.json'  # wie Standard, aber Gradient Cap 1
    config_6 = copy(config)
    config_6.hyper_file = config_6.hyper_file_folder + 'cnn2d_withAddInput_Graph_o1_GlobAtt_o2_2_HO_5.json'  # wie Standard, aber 256,128,64
    config_7 = copy(config)
    config_7.hyper_file = config_7.hyper_file_folder + 'cnn2d_withAddInput_Graph_o1_GlobAtt_o2_2_HO_6.json'  # wie Standard, aber 512,256,128
    config_8 = copy(config)
    config_8.hyper_file = config_8.hyper_file_folder + 'cnn2d_withAddInput_Graph_o1_GlobAtt_o2_2_HO_7.json'  # wie Standard, aber 128,64,32
    config_9 = copy(config)
    config_9.hyper_file = config_9.hyper_file_folder + 'cnn2d_withAddInput_Graph_o1_GlobAtt_o2_2_HO_8.json'  # wie Standard, aber 256,128,128
    config_10 = copy(config)
    config_10.hyper_file = config_10.hyper_file_folder + 'cnn2d_withAddInput_Graph_o1_GlobAtt_o2_2_HO_9.json'  # wie Standard, aber 128,128,128, FC 386-256, CNN2d 128,64,3
    config_11 = copy(config)
    config_11.hyper_file = config_11.hyper_file_folder + 'cnn2d_withAddInput_Graph_o1_GlobAtt_o2_2_HO_10.json'  # wie Standard, aber 128,64,64, FC 386-256, CNN2d 128,64,3
    config_12 = copy(config)
    config_12.hyper_file = config_12.hyper_file_folder + 'cnn2d_withAddInput_Graph_o1_GlobAtt_o2_2_HO_11.json'  # wie Standard, aber 256,128,128 nur mit allem aktiviert
    '''

    '''
    config_3 = copy(config)
    config_3.hyper_file = config_3.hyper_file_folder + 'cnn2d_with_graph_test_Readout_WOowl2vec.json'
    config_4 = copy(config)
    config_4.hyper_file = config_4.hyper_file_folder + 'cnn2d_with_graph_test_Readout_lrSmaller.json'
    config_5 = copy(config)
    config_5.hyper_file = config_5.hyper_file_folder + 'cnn2d_with_graph_test_Readout_WOAttributeWise.json'
    '''
    #list_of_configs = [config_3, config_4, config_5, config_6, config_7, config_8, config_9,config_10, config_11]
    list_of_configs = [config, config_3, config_4]
    #list_of_configs = [config, config_2, config_3,config_4,config_5, config_6,config_7,config_8,config_9,config_10,config_11]

    for i, config in enumerate(list_of_configs):
        print("Run number of config:", i )
        config.print_detailed_config_used_for_training()

        dataset = FullDataset(config.training_data_folder, config, training=True, model_selection=True)
        dataset.load()
        dataset = Representation.convert_dataset_to_baseline_representation(config, dataset)

        checker = ConfigChecker(config, dataset, 'snn', training=True)
        checker.pre_init_checks()

        snn = initialise_snn(config, dataset, True)
        snn.print_detailed_model_info()

        if config.print_model:
            tf.keras.utils.plot_model(snn.encoder.model, to_file='model.png', show_shapes=True, expand_nested=True)

        checker.post_init_checks(snn)

        start_time_string = datetime.now().strftime("%m-%d_%H-%M-%S")

        print('---------------------------------------------')
        print('Training:')
        print('---------------------------------------------')
        print()
        optimizer = SNNOptimizer(snn, dataset, config)
        optimizer.optimize()

        print()
        print('---------------------------------------------')
        print('Evaluation of the current config:')
        print('---------------------------------------------')
        print()
        num_of_selection_tests = config.number_of_selection_tests
        score_valid_to_model_loss = {}
        for i in range(num_of_selection_tests):
            loss_of_selected_model = change_model(config, start_time_string, num_of_selction_iteration=i)

            if config.case_base_for_inference:
                dataset: FullDataset = FullDataset(config.case_base_folder, config, training=False, model_selection=True)
            else:
                dataset: FullDataset = FullDataset(config.training_data_folder, config, training=False, model_selection=True)
            dataset.load()
            dataset = Representation.convert_dataset_to_baseline_representation(config, dataset)

            snn = initialise_snn(config, dataset, False)

            inference = Inference(config, snn, dataset)
            curr_model_score = inference.infer_test_dataset()

            score_valid_to_model_loss[curr_model_score] = loss_of_selected_model

        # loop to sum all values to compute the mean:
        res = 0
        for val in score_valid_to_model_loss.values():
            res += val
        loss_mean = res / len(score_valid_to_model_loss)

        for val in score_valid_to_model_loss.keys():
            res += val
        mean_score = res / len(score_valid_to_model_loss)

        # printing result
        print("Run: ", i, " loss mean:"+ str(loss_mean), " score mean: "+ str(mean_score) )
        print("Run: ", i, " score_valid_to_model_loss:", score_valid_to_model_loss)

        '''
        print()
        print('---------------------------------------------')
        print('Inference:')
        print('---------------------------------------------')
        print()
    
        max_score = max(list(score_valid_to_model_loss.keys()))
        min_loss = score_valid_to_model_loss[max_score]
        print("Model with the following loss is selected for the final evaluation:", min_loss)
    
        change_model(config, start_time_string, get_model_by_loss_value=min_loss)
    
        if config.case_base_for_inference:
            dataset: FullDataset = FullDataset(config.case_base_folder, config, training=False)
        else:
            dataset: FullDataset = FullDataset(config.training_data_folder, config, training=False)
    
        dataset.load()
        dataset = Representation.convert_dataset_to_baseline_representation(config, dataset)
    
        snn = initialise_snn(config, dataset, False)
    
        inference = Inference(config, snn, dataset)
        inference.infer_test_dataset()
        '''


if __name__ == '__main__':
    try:
        for run in range(3):
            print("Experiment ", run, " started!")
            main()
            print("Experiment ", run, " finished!")
    except KeyboardInterrupt:
        pass
