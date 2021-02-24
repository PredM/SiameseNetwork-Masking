import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))


def inverse_dict(dictionary: dict, index):
    keys = dictionary.keys()
    inverse_dict = {}

    for key in keys:
        values = dictionary.get(key)
        inverse_dict[values[index]] = key

    return inverse_dict


def get_attribute_indices(feature_names_all, attribute_matching, index):
    inverse_attribute_matching_1 = inverse_dict(attribute_matching, index)

    ds_attribute_numbers = []

    for index, attribute in enumerate(feature_names_all):
        if attribute in inverse_attribute_matching_1.keys():
            ds_attribute_numbers.append(index)

    return ds_attribute_numbers


def extract(dataset, labels, ds_index, feature_names_all, attribute_matching, failure_mode_matching,
            include_no_failure):
    ds_attribute_indices = get_attribute_indices(feature_names_all, attribute_matching, ds_index)
    dataset_reduced = dataset.copy()[:, :, ds_attribute_indices]

    inverse_class_matching = inverse_dict(failure_mode_matching, ds_index)
    relevant_classes = list(inverse_class_matching.keys())

    if include_no_failure:
        relevant_classes.append('no_failure')

    relevant_example_indices = []
    new_labels = []

    for index, c in enumerate(labels):
        if c in relevant_classes:
            relevant_example_indices.append(index)
            new_label = 'no_failure' if c == 'no_failure' else inverse_class_matching.get(c)
            new_labels.append(new_label)

    dataset_reduced = dataset_reduced[relevant_example_indices, :, :]

    return dataset_reduced, new_labels


def main():
    new_folder = '../data/training_data_transfer/'
    include_no_failure = True

    dataset_folder = '../data/training_data/'

    x_train = np.load(dataset_folder + 'train_features.npy')
    y_train_strings = np.load(dataset_folder + 'train_labels.npy')

    x_test = np.load(dataset_folder + 'test_features.npy')
    y_test_strings = np.load(dataset_folder + 'test_labels.npy')

    feature_names_all = np.load(dataset_folder + 'feature_names.npy')

    # TODO Allow attribute to be in multiple list, currently not working correctly, ensure correct ordering in this case
    # configure matching between attributes
    # key = new attribute name, value = list of transferred attributes
    # first attribute in list should be part of group 1, seconds one of group 2, where group = txt-Module
    attribute_matching = {
        "m_finished": ["txt15_m1.finished", "txt16_m3.finished"],
        "a_x": ["a_15_1_x", "a_16_3_x"],
        "a_y": ["a_15_1_y", "a_16_3_y"],
        "a_z": ["a_15_1_z", "a_16_3_z"],
        "i": ["txt15_i1", "txt16_i4"]
    }

    # configure matching between failure modes, no_failure will always be included
    # Analog to attributes
    failure_mode_matching = {
        "t1_high_wear": ["txt15_m1_t1_high_wear", "txt16_m3_t1_high_wear"],
        "t1_low_wear": ["txt15_m1_t1_low_wear", "txt16_m3_t1_low_wear"],
        "lightbarrier_failure_mode_1": ["txt15_i1_lightbarrier_failure_mode_1", "txt16_i4_lightbarrier_failure_mode_1"]
    }

    new_feature_names = np.array(list(attribute_matching.keys()))
    np.save(new_folder + "ds0/" + "feature_names.npy", new_feature_names)
    np.save(new_folder + "ds1/" + "feature_names.npy", new_feature_names)

    ds0_x_test, ds0_y_test = extract(x_test, y_test_strings, 0, feature_names_all, attribute_matching,
                                     failure_mode_matching, include_no_failure)
    ds0_x_train, ds0_y_train = extract(x_train, y_train_strings, 0, feature_names_all, attribute_matching,
                                       failure_mode_matching, include_no_failure)

    np.save(new_folder + "ds0/" + "test_features.npy", ds0_x_test)
    np.save(new_folder + "ds0/" + "test_labels.npy", ds0_y_test)
    np.save(new_folder + "ds0/" + "train_features.npy", ds0_x_train)
    np.save(new_folder + "ds0/" + "train_labels.npy", ds0_y_train)

    ds1_x_test, ds1_y_test = extract(x_test, y_test_strings, 1, feature_names_all, attribute_matching,
                                     failure_mode_matching, include_no_failure)

    ds1_x_train, ds1_y_train = extract(x_train, y_train_strings, 1, feature_names_all, attribute_matching,
                                       failure_mode_matching, include_no_failure)

    np.save(new_folder + "ds1/" + "test_features.npy", ds1_x_test)
    np.save(new_folder + "ds1/" + "test_labels.npy", ds1_y_test)
    np.save(new_folder + "ds1/" + "train_features.npy", ds1_x_train)
    np.save(new_folder + "ds1/" + "train_labels.npy", ds1_y_train)

    print()
    print("Size overview:")
    print("DS0 train: ", ds0_x_train.shape)
    print("DS0 test: ", ds0_x_test.shape)
    print("DS1 train: ", ds1_x_train.shape)
    print("DS1 test: ", ds1_x_test.shape)


if __name__ == '__main__':
    main()
