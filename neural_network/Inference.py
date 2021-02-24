import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
# suppress debugging messages of TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

from configuration.Enums import SimpleSimilarityMeasure
from neural_network.Evaluator import Evaluator
from configuration.ConfigChecker import ConfigChecker
from configuration.Configuration import Configuration
from neural_network.Dataset import FullDataset
from neural_network.SNN import initialise_snn
from baseline.Representations import Representation


class Inference:

    def __init__(self, config, architecture, dataset: FullDataset):
        self.config: Configuration = config

        self.architecture = architecture
        self.dataset: FullDataset = dataset

        if self.config.inference_with_failures_only:
            self.idx_test_examples_query_pool = self.dataset.get_indices_failures_only_test()
        else:
            self.idx_test_examples_query_pool = range(self.dataset.num_test_instances)

        self.evaluator = Evaluator(dataset, len(self.idx_test_examples_query_pool), self.config.k_of_knn)

    def infer_test_dataset(self):
        start_time = time.perf_counter()

        for idx_test in self.idx_test_examples_query_pool:
            # measure the similarity between the test series and the training batch series
            sims, labels = self.architecture.get_sims(self.dataset.x_test[idx_test])
            # print("sims shape: ", sims.shape, " label shape: ", labels.shape, "self.dataset.x_test.shape", self.dataset.x_test.shape)
            # check similarities of all pairs and record the index of the closest training series

            sims_are_distance_values = True if self.config.simple_measure in [
                SimpleSimilarityMeasure.EUCLIDEAN_DIS] else False
            self.evaluator.add_single_example_results(sims, idx_test, sims_are_distance_values)

        # inference finished
        elapsed_time = time.perf_counter() - start_time

        self.evaluator.calculate_results()
        self.evaluator.print_results(elapsed_time)

        if self.dataset.is_third_party_dataset == False:
            f1_health = self.evaluator.report_dict['no_failure']['f1-score']
            f1_weighted_avg = self.evaluator.report_dict['weighted avg']['f1-score']
            model_score = (f1_health + f1_weighted_avg)/2
        else:
            f1_weighted_avg = self.evaluator.report_dict['weighted avg']['f1-score']
            model_score = f1_weighted_avg
        return model_score


def main():
    config = Configuration()

    if config.case_base_for_inference:
        dataset: FullDataset = FullDataset(config.case_base_folder, config, training=False)
    else:
        dataset: FullDataset = FullDataset(config.training_data_folder, config, training=False)

    dataset.load()
    dataset = Representation.convert_dataset_to_baseline_representation(config, dataset)

    checker = ConfigChecker(config, dataset, 'snn', training=False)
    checker.pre_init_checks()

    architecture = initialise_snn(config, dataset, False)

    checker.post_init_checks(architecture)

    inference = Inference(config, architecture, dataset)

    if config.print_model:
        tf.keras.utils.plot_model(architecture.encoder.model, to_file='model.png', show_shapes=True, expand_nested=True)

    print('Ensure right model file is used:')
    print(config.directory_model_to_use, '\n')

    inference.infer_test_dataset()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
