import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from baseline.Representations import RocketRepresentation
from configuration.Configuration import Configuration
from neural_network.Dataset import FullDataset


def main():
    config = Configuration()

    dataset = FullDataset(config.training_data_folder, config, training=True)
    dataset.load()

    rp: RocketRepresentation = RocketRepresentation(config, dataset)
    rp.create_representation()

    dataset = FullDataset(config.case_base_folder, config, training=True)
    dataset.load()

    rp: RocketRepresentation = RocketRepresentation(config, dataset)
    rp.create_representation(for_case_base=True)

    dataset = FullDataset(config.training_data_folder, config, training=False, model_selection=True)
    dataset.load()

    rp: RocketRepresentation = RocketRepresentation(config, dataset)
    rp.create_representation(for_valid=True)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
