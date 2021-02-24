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


def change_model(config: Configuration, start_time_string):
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

    # Select the best loss and change the config to the corresponding model
    min_loss = min(list(loss_to_dir.keys()))
    config.filename_model_to_use = loss_to_dir.get(min_loss)
    config.directory_model_to_use = config.models_folder + config.filename_model_to_use + '/'

    print('Model selected for inference:')
    print(config.directory_model_to_use, '\n')


# noinspection DuplicatedCode
def main():
    config = Configuration()
    config.print_detailed_config_used_for_training()

    dataset = FullDataset(config.training_data_folder, config, training=True)
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
    print('Inference:')
    print('---------------------------------------------')
    print()
    change_model(config, start_time_string)

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
    try:
        main()
    except KeyboardInterrupt:
        pass
