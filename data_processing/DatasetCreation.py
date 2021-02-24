import gc
import os
import pickle
import sys
import threading

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from data_preprocessing.DatasetCleaning import PostSplitCleaner, PreSplitCleaner
from configuration.Enums import TrainTestSplitMode
from configuration.ConfigChecker import ConfigChecker
from configuration.Configuration import Configuration


class CaseSplitter(threading.Thread):

    def __init__(self, case_info, df: pd.DataFrame):
        super().__init__()
        self.case_label = case_info[0]
        self.start_timestamp_case = case_info[1]
        self.end_timestamp_case = case_info[2]
        self.failure_time = case_info[3]
        self.df = df
        self.result = None

    def run(self):
        try:
            # reassignment is necessary
            case_label = self.case_label
            failure_time = self.failure_time
            start_timestamp_case = self.start_timestamp_case
            end_timestamp_case = self.end_timestamp_case
            df = self.df

            short_label = case_label[0:25] + '...' if len(case_label) > 25 else case_label
            case_info = ['Processing ' + short_label, str(start_timestamp_case), str(end_timestamp_case),
                         'Failure time:', str(failure_time)]

            print("\t{: <50} {: <30} {: <30} {: <20} {: <25}".format(*case_info))

            # basic checks for correct timestamps
            if end_timestamp_case < start_timestamp_case:
                raise KeyError()
            if start_timestamp_case < df.first_valid_index():
                start_timestamp_case = df.first_valid_index()
            if end_timestamp_case > df.last_valid_index():
                end_timestamp_case = df.last_valid_index()

            # extract the part of the case from the dataframe
            self.result = df[start_timestamp_case: end_timestamp_case]

        except KeyError:
            print('CAUTION: Unknown timestamp or wrong order of start/end in at least one case')


# split the dataframe into the failure cases
def split_by_cases(df: pd.DataFrame, data_set_counter, config: Configuration):
    print('\nSplit data by cases with the configured timestamps')

    # get the cases of the dataset after which it should be split
    cases_info = config.cases_datasets[data_set_counter]
    # print(cases_info[1])
    cases = []  # contains dataframes from sensor data
    labels = []  # contains the label of the dataframe
    failures = []  # contains the associated failure time stamp
    threads = []

    # prepare case splitting threads
    for i in range(len(cases_info)):
        t = CaseSplitter(cases_info[i], df)
        threads.append(t)

    # execute threads with the configured amount of parallel threads
    thread_limit = config.max_parallel_cores if len(threads) > config.max_parallel_cores else len(threads)
    threads_finished = 0

    while threads_finished < len(threads):
        if threads_finished + thread_limit > len(threads):
            thread_limit = len(threads) - threads_finished

        r = threads_finished + thread_limit

        print('Processing case', threads_finished, 'to', r - 1)

        for i in range(threads_finished, r):
            threads[i].start()

        for i in range(threads_finished, r):
            threads[i].join()

        for i in range(threads_finished, r):
            if threads[i].result is not None:
                cases.append(threads[i].result)
                labels.append(threads[i].case_label)
                failures.append(threads[i].failure_time)

        threads_finished += thread_limit

    return cases, labels, failures


def extract_single_example(df: pd.DataFrame):
    # No reduction is used if overlapping window is applied
    # because data is down sampled before according parameter sampling frequency
    sampled_values = df.to_numpy()

    # Split sampled values into actual example and values of next timestamp
    example = sampled_values[0:-1]
    next_values = sampled_values[-1]

    # -2 instead of last index because like the version without overlapping time window
    # the last value is not part of the actual example
    time_window_pattern = "%Y-%m-%d %H:%M:%S"
    time_window_string = df.index[0].strftime(time_window_pattern), df.index[-2].strftime(time_window_pattern)

    # print('Example:', example.shape, df.index[0], df.index[0:-1][-1])
    # print('\tNext:', next_values.shape, df.index[-1])

    return example, next_values, time_window_string


def split_into_examples(df: pd.DataFrame, label: str, examples: [np.ndarray], labels_of_examples: [str],
                        config: Configuration,
                        failure_times_of_examples: [str], failure_time,
                        window_times_of_examples: [str], y, i_dataset, next_values: [np.ndarray]):
    start_time = df.index[0]
    end_time = df.index[-1]

    # TODO Check if while is correct: Should be owss or ts_length + freqz
    # slide over data frame and extract windows until the window would exceed the last time step
    while start_time + pd.to_timedelta(config.overlapping_window_step_seconds, unit='s') < end_time:

        # generate a list with indexes for window
        # time_series_length +1 because split the result into actual examples and values of next timestamp
        overlapping_window_indices = pd.date_range(start_time, periods=config.time_series_length + 1,
                                                   freq=config.resample_frequency)

        example, next_values_example, time_window_string = extract_single_example(df.asof(overlapping_window_indices))

        # store information for each example calculated by the threads
        labels_of_examples.append(label)
        examples.append(example)
        next_values.append(next_values_example)
        window_times_of_examples.append(time_window_string)

        # store failure time or special string if no failure example
        if label == 'no_failure':
            failure_times_of_examples.append("noFailure-" + str(i_dataset) + "-" + str(y))
        else:
            failure_times_of_examples.append(str(failure_time))

        # update next start time for next window
        start_time = start_time + pd.to_timedelta(config.overlapping_window_step_seconds, unit='s')


def normalise(x_train: np.ndarray, x_test: np.ndarray, config: Configuration):
    print('\nExecute normalisation')
    length = x_train.shape[2]

    for i in range(length):
        scaler = MinMaxScaler(feature_range=(0, 1))

        # reshape column vector over each example and timestamp to a flatt array
        # necessary for normalisation to work properly
        shape_before = x_train[:, :, i].shape
        x_train_shaped = x_train[:, :, i].reshape(shape_before[0] * shape_before[1], 1)

        # learn scaler only on training data (best practice)
        x_train_shaped = scaler.fit_transform(x_train_shaped)

        # reshape back to original shape and assign normalised values
        x_train[:, :, i] = x_train_shaped.reshape(shape_before)

        # normalise test data
        shape_before = x_test[:, :, i].shape
        x_test_shaped = x_test[:, :, i].reshape(shape_before[0] * shape_before[1], 1)
        x_test_shaped = scaler.transform(x_test_shaped)
        x_test[:, :, i] = x_test_shaped.reshape(shape_before)

        # export scaler to use with live data
        scaler_filename = config.scaler_folder + 'scaler_' + str(i) + '.save'
        joblib.dump(scaler, scaler_filename)

    return x_train, x_test


def determine_train_test_indices(config: Configuration, examples_array, labels_array, failure_times_array):
    if config.split_mode == TrainTestSplitMode.ENSURE_NO_MIX:

        # Split into train and test considering that examples from a single failure run don't end up in both
        print('\nExecute train/test split in ENSURE_NO_MIX mode.')
        enc = OrdinalEncoder()
        enc.fit(failure_times_array.reshape(-1, 1))
        failure_times_array_groups = enc.transform(failure_times_array.reshape(-1, 1))

        gss = GroupShuffleSplit(n_splits=1, test_size=config.test_split_size, random_state=config.random_seed)

        train_indices, test_indices = list(gss.split(examples_array, labels_array, failure_times_array_groups))[0]

    elif config.split_mode == TrainTestSplitMode.ANOMALY_DETECTION:

        # This means all failure examples are in test
        # Only no_failure examples will be split based on configured percentage
        print('\nExecute train/test split in ANOMALY_DETECTION mode.')

        # Split examples into normal and failure cases
        failure_indices = np.argwhere(labels_array != 'no_failure').flatten()
        no_failure_indices = np.argwhere(labels_array == 'no_failure').flatten()

        # Execute recording instance based splitting only for no_failures
        # For which the input arrays are first of all reduced to those examples
        nf_examples = examples_array[no_failure_indices]
        nf_labels = labels_array[no_failure_indices]
        nf_failure_times = failure_times_array[no_failure_indices]

        enc = OrdinalEncoder()
        enc.fit(nf_failure_times.reshape(-1, 1))
        nf_groups = enc.transform(nf_failure_times.reshape(-1, 1))

        # Split the no failure only examples based on the recording instances and the split size
        gss = GroupShuffleSplit(n_splits=1, test_size=config.test_split_size, random_state=config.random_seed)
        nf_train_indices_in_reduced, nf_test_indices_in_reduced = \
            list(gss.split(nf_examples, nf_labels, nf_groups))[0]

        # Trace back the indices of the reduced arrays to the indices of the complete arrays
        nf_train_indices = no_failure_indices[nf_train_indices_in_reduced]
        nf_test_indices = no_failure_indices[nf_test_indices_in_reduced]

        # Combine indices to full lists
        # Train part only consists of the  train part of the no failure split,
        # whereas the test part consists of the test part of the no failure split as well as failure examples
        train_indices = list(nf_train_indices)
        test_indices = list(failure_indices) + list(nf_test_indices)
    else:
        raise ValueError()

    return train_indices, test_indices


def main():
    config = Configuration()  # Get config for data directory

    checker = ConfigChecker(config, None, 'preprocessing', training=None)
    checker.pre_init_checks()

    config.import_timestamps()
    number_data_sets = len(config.datasets)

    # list of all examples
    examples: [np.ndarray] = []
    labels_of_examples: [str] = []
    next_values: [np.ndarray] = []
    failure_times_of_examples: [str] = []
    window_times_of_examples: [str] = []

    attributes = None

    for i in range(number_data_sets):
        print('\n\nImporting dataframe ' + str(i) + '/' + str(number_data_sets - 1) + ' from file')

        # read the imported dataframe from the saved file
        path_to_file = config.datasets[i][0] + config.filename_pkl_cleaned

        with open(path_to_file, 'rb') as f:
            df: pd.DataFrame = pickle.load(f)

        # split the dataframe into the configured cases
        cases_df, labels_df, failures_df = split_by_cases(df, i, config)

        if i == 0:
            attributes = np.stack(df.columns, axis=0)

        del df
        gc.collect()

        # split the case into examples, which are added to the list of of all examples
        number_cases = len(cases_df)

        print()
        for y in range(number_cases):
            df = cases_df[y]

            if len(df) <= 0:
                print(i, y, ' is empty!')
                continue

            start = df.index[0]
            end = df.index[-1]
            secs = (end - start).total_seconds()
            print('Splitting case', y, '/', number_cases - 1, 'into examples. Length:', secs, " Start: ", start,
                  " End: ", end)
            split_into_examples(df, labels_df[y], examples, labels_of_examples, config,
                                failure_times_of_examples, failures_df[y],
                                window_times_of_examples, y, i, next_values)
        print()
        del cases_df, labels_df, failures_df
        gc.collect()

    # convert lists of arrays to numpy array
    examples_array = np.stack(examples, axis=0)
    labels_array = np.stack(labels_of_examples, axis=0)
    next_values_array = np.stack(next_values, axis=0)
    failure_times_array = np.stack(failure_times_of_examples, axis=0)
    window_times_array = np.stack(window_times_of_examples, axis=0)

    del examples, labels_of_examples, failure_times_of_examples, window_times_of_examples
    gc.collect()

    cleaner = PreSplitCleaner(config, examples_array, labels_array, next_values_array, failure_times_array,
                              window_times_array)

    print('\nExamples before pre train/test split cleaning:', examples_array.shape[0])
    cleaner.clean()
    examples_array, labels_array, next_values_array, failure_times_array, window_times_array = cleaner.return_all()
    print('Examples after pre train/test split cleaning:', examples_array.shape[0])

    train_indices, test_indices = determine_train_test_indices(config, examples_array, labels_array,
                                                               failure_times_array)

    x_train, x_test = examples_array[train_indices], examples_array[test_indices]
    y_train, y_test = labels_array[train_indices], labels_array[test_indices]
    next_values_train, next_values_test = next_values_array[train_indices], next_values_array[test_indices]
    failure_times_train, failure_times_test = failure_times_array[train_indices], failure_times_array[test_indices]
    window_times_train, window_times_test = window_times_array[train_indices], window_times_array[test_indices]

    del examples_array, labels_array, next_values_array, failure_times_array, window_times_array
    gc.collect()

    # Execute some manual corrections
    cleaner = PostSplitCleaner(config,
                               x_train, x_test,
                               y_train, y_test,
                               next_values_train, next_values_test,
                               failure_times_train, failure_times_test,
                               window_times_train, window_times_test)

    print('\nExamples in train before:', x_train.shape[0])
    print('Examples in test before:', x_test.shape[0], '\n')

    cleaner.clean()

    x_train, x_test, y_train, y_test, next_values_train, next_values_test, \
    failure_times_train, failure_times_test, window_times_train, window_times_test = cleaner.return_all()

    print('\nExamples in train after:', x_train.shape[0])
    print('Examples in test after:', x_test.shape[0], '\n')

    print("x_train:", x_train.shape, "x_test:", x_test.shape)
    print("y_train:", y_train.shape, "y_test:", y_test.shape)
    print("next_values_train:", next_values_train.shape, "next_values_test:", next_values_test.shape)
    print("failure_times_train:", failure_times_train.shape, "failure_times_test:", failure_times_test.shape)
    print("window_times_train:", window_times_train.shape, "window_times_test:", window_times_test.shape)
    print()
    print("Classes in the train set:\n", np.unique(y_train))
    print("Classes in the test set:\n", np.unique(y_test))

    # normalize each sensor stream to contain values in [0,1]
    x_train, x_test = normalise(x_train, x_test, config)

    # cast to float32 so it can directly be used as tensorflow input without casting
    x_train, x_test, = x_train.astype('float32'), x_test.astype('float32')

    # save the np arrays
    print('\nSave to np arrays in ' + config.training_data_folder)

    print('Step 1/5')
    np.save(config.training_data_folder + 'train_features.npy', x_train)
    print('Step 2/5')
    np.save(config.training_data_folder + 'test_features.npy', x_test)
    print('Step 3/5')
    np.save(config.training_data_folder + 'train_labels.npy', y_train)
    print('Step 4/5')
    np.save(config.training_data_folder + 'test_labels.npy', y_test)
    print('Step 5/5')
    np.save(config.training_data_folder + 'feature_names.npy', attributes)
    print()

    print('Saving additional data')

    # Contains the associated time of a failure (if not no failure) for each example
    print('Step 1/6')
    np.save(config.training_data_folder + 'train_failure_times.npy', failure_times_train)
    print('Step 2/6')
    np.save(config.training_data_folder + 'test_failure_times.npy', failure_times_test)

    # Contains the start and end time stamp for each training example
    print('Step 3/6')
    np.save(config.training_data_folder + 'train_window_times.npy', window_times_train)
    print('Step 4/4')
    np.save(config.training_data_folder + 'test_window_times.npy', window_times_test)

    # Contain the values of the next timestamp after each training example
    print('Step 5/6')
    np.save(config.training_data_folder + 'train_next_values.npy', next_values_train)
    print('Step 6/6')
    np.save(config.training_data_folder + 'test_next_values.npy', next_values_test)
    print()


if __name__ == '__main__':
    main()
