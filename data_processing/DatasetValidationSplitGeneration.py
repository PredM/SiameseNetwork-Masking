import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))


from configuration.Configuration import Configuration

def main():
    np.random.seed(2021)
    # import data sets
    # 4 sec windows (overlapping) with 4ms sampling
    config = Configuration()
    x_train_features = np.load(config.training_data_folder + "train_features.npy")  # data streams to train a machine learning model
    x_test_features = np.load(config.training_data_folder +"test_features.npy")  # data streams to for test a machine learning model
    y_train = np.load(config.training_data_folder +"train_labels.npy")  # labels of the training data
    y_test = np.load(config.training_data_folder +"test_labels.npy")  # labels of the training data
    train_window_times = np.load(config.training_data_folder +"train_window_times.npy")  # labels of the training data
    test_window_times = np.load(config.training_data_folder +"test_window_times.npy")  # labels of the training data
    train_failure_times = np.load(config.training_data_folder +"train_failure_times.npy")  # labels of the training data
    test_failure_times = np.load(config.training_data_folder +"test_failure_times.npy")  # labels of the training data

    feature_names = np.load(config.training_data_folder +"feature_names.npy")

    print("Train shape: ", x_train_features.shape)
    print("feature_names: ", feature_names)
    print("y_train: ", y_train.shape)

    # Number of failure examples per class
    k = 1
    print("k is set to: ",k)
    #                                       Train       Test    Ratio
    # NoFailure examples in train, test:    24.908,     2.907   0.11670949092
    # FaF examples in train, test:          642,        482     0.75077881619
    # Ratio                                 0.02577485145 0.16580667354

    # Valid: 0.5% NoFailure Examples: 125
    # Valid: 0.5% FaF       Examples: 3

    num_NoFailre_examples = 330             #190
    num_FaF_examples_perClass_small = 2     #2
    num_FaF_examples_perClass_big = 3       #3
    num_min_FaF_examples_per_class =  8      #8
    # current ratio:                        # 55/ 189

    # Get a unique list of labels
    y_train_unique = np.unique(y_train)
    remove_examples_idx = None
    remove_examples_idx_no_failure = np.empty((0,), int)

    # Iterate over each label and extract according previously given restrictions
    for i, label in enumerate(y_train_unique):
            # Get idx of examples with this label
            example_idx_of_curr_label = np.where(y_train == label)
            # Prepare input to a 1d array for choice method
            example_idx_of_curr_label = np.squeeze(example_idx_of_curr_label)

            # Select the indexes of examples that should be used for the validation set according the given restrictions for each class
            if label == "no_failure":
                # Select num_NoFailre_examples examples randomly
                k_examples_of_curr_label = np.random.choice(example_idx_of_curr_label,num_NoFailre_examples)
                k = num_NoFailre_examples
                # Get 3 sec before and 3 sec after each example
                for curr_idx in k_examples_of_curr_label:
                    remove_examples_idx_no_failure = np.append(remove_examples_idx_no_failure, curr_idx+1)
                    remove_examples_idx_no_failure = np.append(remove_examples_idx_no_failure, curr_idx + 2)
                    remove_examples_idx_no_failure = np.append(remove_examples_idx_no_failure, curr_idx + 3)

                    remove_examples_idx_no_failure = np.append(remove_examples_idx_no_failure, curr_idx-1)
                    remove_examples_idx_no_failure = np.append(remove_examples_idx_no_failure, curr_idx - 2)
                    remove_examples_idx_no_failure = np.append(remove_examples_idx_no_failure, curr_idx - 3)
            else:
                if example_idx_of_curr_label.shape[0] >= num_min_FaF_examples_per_class:
                    if label in ["txt15_m1_t1_high_wear","txt15_m1_t1_low_wear", "txt15_m1_t2_wear", "txt16_m3_t2_wear",
                    "txt16_m3_t1_high_wear", "txt18_pneumatic_leakage_failure_mode_2", "txt17_pneumatic_leakage_failure_mode_1",
                    "txt17_i1_switch_failure_mode_2", "txt17_workingstation_transport_failure_mode_wout_workpiece",
                    "txt16_i4_lightbarrier_failure_mode_1"]:
                        k_examples_of_curr_label = np.random.choice(example_idx_of_curr_label,
                                                                    num_FaF_examples_perClass_big)
                        k = num_FaF_examples_perClass_big
                    else:
                        k_examples_of_curr_label = np.random.choice(example_idx_of_curr_label, num_FaF_examples_perClass_small)
                        k = num_FaF_examples_perClass_small
                else:
                    print("Class ", label , "does ony have ", example_idx_of_curr_label.shape[0], " examples and for this reason is excluded." )
                    k_examples_of_curr_label = np.random.choice(example_idx_of_curr_label, 0)

            #Store a list with examples that should be removed from train and used as validation set
            if i == 0:
                remove_examples_idx = k_examples_of_curr_label
            else:
                remove_examples_idx = np.append(remove_examples_idx, k_examples_of_curr_label)

            print("Label: ", label, " has ", example_idx_of_curr_label.shape[0], " training examples from which the following ",
              k, " are chosen: ", k_examples_of_curr_label)

    print("Examples for using as a validation set are randomy chosen according given restrictions")
    #Create masking for generating the new training data
    mask = np.isin(np.arange(x_train_features.shape[0]), remove_examples_idx)
    #print("mask shape: ", mask.shape)
    #print("np.arange(x_train_features.shape[0]): ", np.arange(x_train_features.shape[0]))
    remove_mask = np.isin(np.arange(x_train_features.shape[0]), remove_examples_idx, invert=True)

    # Create train set ( that has a validation set removed)
    x_train_features_new_train = x_train_features[remove_mask,:,:]
    y_train_new_train = y_train[remove_mask]
    train_window_times_new_train = train_window_times[remove_mask]
    train_failure_times_new_train = train_failure_times[remove_mask]

    # Remove no_failure examples with overlapping windows
    remove_mask_no_failure_overlapping_windows = np.isin(np.arange(x_train_features_new_train.shape[0]), remove_examples_idx_no_failure, invert=True)

    x_train_features_new_train = x_train_features_new_train[remove_mask_no_failure_overlapping_windows,:,:]
    y_train_new_train = y_train_new_train[remove_mask_no_failure_overlapping_windows]
    train_window_times_new_train = train_window_times_new_train[remove_mask_no_failure_overlapping_windows]
    train_failure_times_new_train = train_failure_times_new_train[remove_mask_no_failure_overlapping_windows]

    # Create validation data set
    # 1-mask flips zeros to ones and ones to zeros
    x_train_features_new_valid = x_train_features[mask,:,:]
    y_train_new_valid = y_train[mask]
    train_window_times_new_valid = train_window_times[mask]
    train_failure_times_new_valid = train_failure_times[mask]

    print("x_train_features_new_train: ", x_train_features_new_train.shape)
    print("y_train_new_train: ", y_train_new_train.shape)
    print("train_window_times_new_train: ", train_window_times_new_train.shape)
    print("train_failure_times_new_train: ", train_failure_times_new_train.shape)

    print("x_train_features_new_valid: ", x_train_features_new_valid.shape)
    print("y_train_new_valid: ", y_train_new_valid.shape)
    print("train_window_times_new_valid: ", train_window_times_new_valid.shape)
    print("train_failure_times_new_valid: ", train_failure_times_new_valid.shape)

    # save the modified data
    print('\nSave  to np arrays in ' + config.training_data_folder)
    print('Step 1/4 train_features_new_train')
    np.save(config.training_data_folder + 'train_features_new2.npy', x_train_features_new_train)
    print('Step 2/4 train_labels_new_train')
    np.save(config.training_data_folder + 'train_labels_new2.npy', y_train_new_train)
    print('Step 3/4 train_window_times_new_train')
    np.save(config.training_data_folder + 'train_window_times_new2.npy', train_window_times_new_train)
    print('Step 4/4 train_failure_times_new_train')
    np.save(config.training_data_folder + 'train_failure_times_new2.npy', train_failure_times_new_train)

    print('Step 1/4 valid_features_new_valid')
    np.save(config.training_data_folder + 'valid_features_new2.npy', x_train_features_new_valid)
    print('Step 2/4 valid_labels_new_valid')
    np.save(config.training_data_folder + 'valid_labels_new2.npy', y_train_new_valid)
    print('Step 3/4 valid_window_times_new_valid')
    np.save(config.training_data_folder + 'valid_window_times_new2.npy', train_window_times_new_valid)
    print('Step 4/4 valid_failure_times_new_train')
    np.save(config.training_data_folder + 'valid_failure_times_new2.npy', train_failure_times_new_valid)

if __name__ == '__main__':
    main()
