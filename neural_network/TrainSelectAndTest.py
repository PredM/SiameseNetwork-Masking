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

        selected_loss = loss_list[num_of_selction_iteration]

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
    print('Selecting (of the model for final evaluation):')
    print('---------------------------------------------')
    print()
    num_of_selection_tests = config.number_of_selection_tests
    config.use_masking_regularization = False
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

    print("score_valid_to_model_loss: ", score_valid_to_model_loss)

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


if __name__ == '__main__':
    num_of_runs = 3
    try:
        for run in range(num_of_runs):
            print("Experiment ", run, " started!")
            main()
            print("Experiment ", run, " finished!")
    except KeyboardInterrupt:
        pass
