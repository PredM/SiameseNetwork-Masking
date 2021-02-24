import math

import numpy as np

from configuration.Configuration import Configuration
from configuration.Enums import BatchSubsetType, LossFunction
from configuration.Hyperparameter import Hyperparameters
from neural_network.Dataset import FullDataset, CBSDataset


class BatchComposer:

    def __init__(self, config, dataset, hyper, from_test):
        self.config: Configuration = config
        self.dataset: FullDataset = dataset
        self.hyper: Hyperparameters = hyper

        self.y = self.dataset.y_test if from_test else self.dataset.y_train
        self.labels = self.dataset.y_test_strings_unique if from_test else self.dataset.y_train_strings_unique
        self.num_instances = self.dataset.num_test_instances if from_test else self.dataset.num_train_instances
        self.mapping = self.dataset.class_idx_to_ex_idxs_test if from_test else self.dataset.class_idx_to_ex_idxs_train
        self.failure_indices_only = []

        for key in self.mapping.keys():
            if not key == 'no_failure':
                self.failure_indices_only.append(self.mapping.get(key))

        self.failure_indices_only = np.sort(np.concatenate(self.failure_indices_only))

    def compose_batch(self):
        batch_true_similarities = []  # similarity label for each pair
        batch_pair_indices = []  # index number of each example used in the training

        for subset_type in self.config.batch_distribution.keys():
            percentage_type = self.config.batch_distribution.get(subset_type)
            # Ceil rounds up to next integer
            nbr_pairs_with_this_type = math.ceil(percentage_type * self.hyper.batch_size)

            if self.config.type_of_loss_function == LossFunction.TRIPLET_LOSS:
                subset_pair_indices, subset_true_similarities = self.compose_triplet_subset(subset_type,
                                                                                            nbr_pairs_with_this_type)
            else:
                subset_pair_indices, subset_true_similarities = self.compose_subset(subset_type,
                                                                                    nbr_pairs_with_this_type)

            batch_pair_indices.extend(subset_pair_indices)
            batch_true_similarities.extend(subset_true_similarities)

        # Change the list of ground truth similarities to an array
        true_similarities = np.asarray(batch_true_similarities)

        return batch_pair_indices, true_similarities

    def compose_subset(self, subset_type, nbr_pairs):
        subset_pair_indices, subset_true_similarities = [], []

        # // 2 because each iteration one similar and one dissimilar pair is added
        for i in range(nbr_pairs // 2):
            # Pos pairs
            pos_pair = self.draw_pair(is_positive=True, type=subset_type)
            subset_pair_indices.append(pos_pair[0])
            subset_pair_indices.append(pos_pair[1])
            subset_true_similarities.append(1.0)

            # Neg pairs
            neg_pair = self.draw_pair(is_positive=False, type=subset_type)
            subset_pair_indices.append(neg_pair[0])
            subset_pair_indices.append(neg_pair[1])

            # If configured a similarity value is used for the negative pair instead of full dissimilarity
            if self.config.use_sim_value_for_neg_pair:
                sim = self.dataset.get_sim_label_pair(neg_pair[0], neg_pair[1], 'train')
                subset_true_similarities.append(sim)
            else:
                subset_true_similarities.append(0.0)

        return subset_pair_indices, subset_true_similarities

    def draw_pair(self, is_positive, type=None):

        if type == BatchSubsetType.DISTRIB_BASED_ON_DATASET:
            # Compared to old implementation we keep the first index and only redraw the second one
            # in order to maybe increase the chance of non no failure pairs
            first_idx = np.random.randint(0, self.num_instances, size=1)[0]

            # This way of determining a second index for a positive pair is faster than looping
            if is_positive:
                class_of_first = np.nonzero(self.y[first_idx] == 1)[0][0]
                examples_with_same = self.mapping.get(self.dataset.one_hot_index_to_string.get(class_of_first))
                second_idx = np.random.choice(examples_with_same, 1)[0]
                return first_idx, second_idx
            else:
                while True:
                    second_idx = np.random.randint(0, self.num_instances, size=1)[0]
                    # return if pair matches the is_positive criterion, else draw another one
                    if not np.array_equal(self.y[first_idx], self.y[second_idx]):
                        return first_idx, second_idx

        elif type == BatchSubsetType.EQUAL_CLASS_DISTRIB:

            if is_positive:
                class_both_idx = np.random.choice(self.labels, size=1)[0]
                return np.random.choice(self.mapping.get(class_both_idx), 2, replace=True)

            else:
                # Replace = False ensures we get two different classes
                class_first_idx, class_second_idx = np.random.choice(self.labels, size=2, replace=False)
                first_idx = np.random.choice(self.mapping.get(class_first_idx), 1)[0]
                second_idx = np.random.choice(self.mapping.get(class_second_idx), 1)[0]
                return first_idx, second_idx

        elif type == BatchSubsetType.ONLY_FAILURE_PAIRS or type == BatchSubsetType.NO_FAILURE_ONLY_FOR_NEG_PAIRS:
            first_idx = np.random.choice(self.failure_indices_only, 1)[0]

            # Equivalent to the 1st variant, but it is only drawn from the error cases
            if is_positive:
                class_of_first = np.nonzero(self.y[first_idx] == 1)[0][0]
                examples_with_same = self.mapping.get(self.dataset.one_hot_index_to_string.get(class_of_first))
                second_idx = np.random.choice(examples_with_same, 1)[0]
                return first_idx, second_idx
            else:
                while True:

                    # Second index for negative pair is drawn based type
                    if type == BatchSubsetType.ONLY_FAILURE_PAIRS:
                        second_idx = np.random.choice(self.failure_indices_only, 1)[0]
                    else:
                        second_idx = np.random.randint(0, self.num_instances, size=1)[0]

                    # Return if pair matches the is_positive criterion, else draw another one
                    if not np.array_equal(self.y[first_idx], self.y[second_idx]):
                        return first_idx, second_idx
        else:
            raise NotImplementedError('Unknown batch subset type:', type)

    def compose_triplet_subset(self, subset_type, nbr_pairs):
        # subset_true_similarities are not used so any empty list can be returned in order so be compatible with the
        # compose_batch method
        subset_pair_indices, subset_true_similarities = [], []

        for i in range(nbr_pairs // 2):
            # Pos pairs
            triplet = self.draw_triplet(subset_type)
            subset_pair_indices.append(triplet[0])
            subset_pair_indices.append(triplet[1])
            subset_pair_indices.append(triplet[0])
            subset_pair_indices.append(triplet[2])

        return subset_pair_indices, subset_true_similarities

    # Distribution of drawn examples based on dataset
    def draw_triplet(self, type=None):

        if type == BatchSubsetType.DISTRIB_BASED_ON_DATASET:

            # Get comparison example and its class
            i = np.random.randint(0, self.num_instances, size=1)[0]
            class_of_i = np.nonzero(self.y[i] == 1)[0][0]

            # Get positive example
            examples_with_same = self.mapping.get(self.dataset.one_hot_index_to_string.get(class_of_i))
            j = np.random.choice(examples_with_same, 1)[0]

            # Get negative example and return all
            while True:
                l = np.random.randint(0, self.num_instances, size=1)[0]
                # return if pair matches the is_positive criterion, else draw another one
                if not np.array_equal(self.y[j], self.y[l]):
                    return i, j, l

        elif type == BatchSubsetType.EQUAL_CLASS_DISTRIB:
            # Replace = False ensures we get two different classes
            class_i_and_j, class_l = np.random.choice(self.labels, size=2, replace=False)
            i, j = np.random.choice(self.mapping.get(class_i_and_j), 2, replace=True)
            l = np.random.choice(self.mapping.get(class_l), 1)[0]
            return i, j, l

        else:
            raise NotImplementedError('Unknown batch subset type for triplet loss:', type)


# TODO Maybe integrate in SNN BatchComposer
#  if group id == none SNN behaviour else cbs behaviour
# TODO Currently fixed to train (not test) dataset
class CbsBatchComposer(BatchComposer):
    def __init__(self, config, dataset, hyper, from_test, group_id):
        super().__init__(config, dataset, hyper, from_test)
        self.group_id = group_id
        self.dataset: CBSDataset = dataset

        # Get the classes the handler of with this id is responsible for and ensure only those are in the list
        # that are also contained in the dataset
        self.relevant_classes = list(set(self.config.group_id_to_cases.get(self.group_id)).intersection(
            set(self.dataset.classes_total)))


    def draw_pair(self, is_positive, type=None):

        if type == BatchSubsetType.DISTRIB_BASED_ON_DATASET:
            pos_indices = self.dataset.group_to_indices_train.get(self.group_id)
            first_idx = np.random.choice(pos_indices, 1, replace=True)[0]

            # This way of determining a second index for a positive pair is faster than looping
            if is_positive:
                class_of_first = np.nonzero(self.y[first_idx] == 1)[0][0]
                examples_with_same = self.mapping.get(self.dataset.one_hot_index_to_string.get(class_of_first))
                second_idx = np.random.choice(examples_with_same, 1)[0]

                return first_idx, second_idx
            else:
                while True:
                    second_idx = np.random.randint(0, self.num_instances, size=1)[0]
                    # return if pair matches the is_positive criterion, else draw another one
                    if not np.array_equal(self.y[first_idx], self.y[second_idx]):
                        return first_idx, second_idx

        elif type == BatchSubsetType.EQUAL_CLASS_DISTRIB:
            class_first_idx = np.random.choice(self.relevant_classes, 1, replace=True)[0]

            if class_first_idx not in self.dataset.classes_total:
                print(self.group_id)
                print(self.relevant_classes)
                raise ValueError('Class in config.json does not match the dataet:' + class_first_idx)

            if is_positive:
                return np.random.choice(self.mapping.get(class_first_idx), 2, replace=True)
            else:
                while True:
                    class_second_idx = np.random.choice(self.labels, size=1, replace=False)[0]

                    if class_second_idx not in self.dataset.classes_total:
                        print(self.group_id)
                        print(self.relevant_classes)
                        raise ValueError('Class in config.json does not match the dataet:' + class_first_idx)

                    if class_first_idx != class_second_idx:
                        first_idx = np.random.choice(self.mapping.get(class_first_idx), 1)[0]
                        second_idx = np.random.choice(self.mapping.get(class_second_idx), 1)[0]
                        return first_idx, second_idx

        else:
            raise NotImplementedError('Subset type not implemented for CBS: ', type)
