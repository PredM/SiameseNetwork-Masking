import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

# suppress debugging messages of TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from configuration.ConfigChecker import ConfigChecker
from configuration.Configuration import Configuration
from neural_network.Dataset import FullDataset
from neural_network.Optimizer import SNNOptimizer
from neural_network.SNN import initialise_snn
from baseline.Representations import Representation


def main():
    config = Configuration()

    dataset = FullDataset(config.training_data_folder, config, training=True)
    dataset.load()
    dataset = Representation.convert_dataset_to_baseline_representation(config, dataset)

    checker = ConfigChecker(config, dataset, 'snn', training=True)
    checker.pre_init_checks()

    snn = initialise_snn(config, dataset, True)
    snn.print_detailed_model_info()

    checker.post_init_checks(snn)

    print('Training:')
    optimizer = SNNOptimizer(snn, dataset, config)
    optimizer.optimize()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
