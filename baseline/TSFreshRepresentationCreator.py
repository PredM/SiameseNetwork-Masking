import os
import sys
import psutil

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from baseline.Representations import TSFreshRepresentation
from configuration.Configuration import Configuration
from neural_network.Dataset import FullDataset


def main():
    config = Configuration()

    p = psutil.Process()
    cores = p.cpu_affinity()
    p.cpu_affinity(cores[0:config.max_parallel_cores])

    # Feature extraction for full dataset is disabled by default because of high computational demands

    # dataset = FullDataset(config.training_data_folder, config, training=True)
    # dataset.load()
    # rp: TSFreshRepresentation = TSFreshRepresentation(config, dataset)
    # rp.create_representation()

    dataset = FullDataset(config.case_base_folder, config, training=True)
    dataset.load()

    rp: TSFreshRepresentation = TSFreshRepresentation(config, dataset)
    rp.create_representation(for_case_base=True)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
