import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

import joblib
import pandas as pd
import numpy as np

from enum import Enum
from math import ceil
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GroupShuffleSplit


class SplitVariant(Enum):
    # Use the split implemented by the dataset authors (only contains no_failure in train)
    PREDEFINED = 0

    # Use total random split by sklearn
    RANDOM_SPLIT = 1

    # Split based on date_threshold defined below: all instances recorded before this date will be in train
    DATE_BASED = 2

    # ensures examples from same failure occurrence (assumed: same date) won't end up both in train and test
    ENSURE_NO_MIX = 3


#######################################################################################################################

working_directory = Path('../../..', 'data/additional_datasets/3w_dataset/')

events_names = {0: 'Normal',
                1: 'Abrupt Increase of BSW',
                2: 'Spurious Closure of DHSV',
                3: 'Severe Slugging',
                4: 'Flow Instability',
                5: 'Rapid Productivity Loss',
                6: 'Quick Restriction in PCK',
                7: 'Scaling in PCK',
                8: 'Hydrate in Production Line'
                }

vars = ['P-PDG',
        'P-TPT',
        'T-TPT',
        'P-MON-CKP',
        'T-JUS-CKP',
        'P-JUS-CKGL',
        'T-JUS-CKGL',
        'QGL']

columns = ['timestamp'] + vars + ['class']
normal_class_code = 0
abnormal_classes_codes = [1, 2, 5, 6, 7, 8]
min_normal_period_size = 20 * 60  # In observations = seconds
max_nan_percent = 0.1  # For selection of useful variables
std_vars_min = 0.01  # For selection of useful variables
disable_progressbar = True  # For less output

# TODO evtl. noch Anpassungen vornehmen
split_range = 0.8  # Changed to match config setting, % of examples in train set
split_random_seed = 23
max_samples_per_period = 15  # Limitation for safety # TODO Erhöhung möglich/sinnvoll?
sample_size = 3 * 60  # In observations = seconds

# New settings
split_variant = SplitVariant.ENSURE_NO_MIX
date_threshold = '31-10-2017'


#######################################################################################################################
def replace_label(input_array, target_label, new_label):
    new_labels = []

    for label in input_array:
        if label == target_label:
            new_labels.append(new_label)
        else:
            new_labels.append(label)

    return np.array(new_labels)


def main():
    pd.set_option('display.max_rows', 500)

    print('\nImporting instances ...')
    real_instances = pd.DataFrame(
        class_and_file_generator(working_directory.joinpath('datasets/'), real=True, simulated=False, drawn=False),
        columns=['class_code', 'instance_path'])

    # We also want no failure cases -> No filtering here
    # real_instances = real_instances.loc[real_instances.iloc[:, 0].isin(abnormal_classes_codes)].reset_index(drop=True)

    x_train_dfs = []
    x_test_dfs = []
    y_train_lists = []
    y_test_lists = []

    print('\nExtracting single instances ...')
    ignored_instances = 0
    used_instances = 0
    for i, row in real_instances.iterrows():

        # Loads the current instance
        class_code, instance_path = row
        print('instance {}: {}'.format(i + 1, instance_path))
        df = load_instance(instance_path)

        # Ignores instances without sufficient normal periods
        normal_period_size = (df['class'] == float(normal_class_code)).sum()
        if normal_period_size < min_normal_period_size:
            ignored_instances += 1
            print('\tskipped because normal_period_size is insufficient for training ({})'.format(normal_period_size))
            continue
        used_instances += 1

        # Extracts samples from the current real instance
        ret = extract_samples(df, class_code)
        df_samples_train, y_train, df_samples_test, y_test = ret

        # We don't want a only binary classification
        # y_test[y_test != normal_class_code] = -1
        # y_test[y_test == normal_class_code] = 1

        x_train_dfs.append(df_samples_train)
        x_test_dfs.append(df_samples_test)
        y_train_lists.append(y_train)
        y_test_lists.append(y_test)

    # Adaptation of the ID of the individual examples so that they are not mixed up later when grouped by ID
    # --> Ensures that the IDs are unique across all examples, not just per DF
    counter = 0
    for df in x_train_dfs:
        examples_in_df = df['id'].max()
        df['id'] = df['id'] + counter
        counter += examples_in_df + 1

    for df in x_test_dfs:
        examples_in_df = df['id'].max()
        df['id'] = df['id'] + counter
        counter += examples_in_df + 1

    df_train_combined = pd.concat(x_train_dfs)
    df_test_combined = pd.concat(x_test_dfs)

    # Series of how many nan there are per attribute
    nans_in_train = df_train_combined.isnull().sum()
    nans_in_test = df_test_combined.isnull().sum()

    if not ((nans_in_test == 0).all() and (nans_in_train == 0).all()):
        raise Exception('NaN value found - handling not implemented yet')
    else:
        print('\nNo NaN values found.')

    print('\nCombining into single numpy array...')

    # must be done here before grouping
    attribute_names = df_train_combined.columns.values

    df_train_combined = df_train_combined.groupby('id')
    df_test_combined = df_test_combined.groupby('id')

    dates_train = pd.to_datetime(df_train_combined.first()['timestamp']).dt.strftime('%d-%m-%Y').values
    dates_test = pd.to_datetime(df_test_combined.first()['timestamp']).dt.strftime('%d-%m-%Y').values

    x_train = np.array(list(df_train_combined.apply(pd.DataFrame.to_numpy)))
    x_test = np.array(list(df_test_combined.apply(pd.DataFrame.to_numpy)))

    y_train = np.array(y_train_lists).flatten().astype(str)
    y_test = np.array(y_test_lists).flatten().astype(str)

    # TODO Replace numbers with class names
    #  Current Problem: Why are there classes 106, 107? Or: why aren't they in the dict above?
    #  Maybe 106 is a special type of 6 - Check in paper
    # print(y_train[[0, 1, 2, 3, 4, -2, -1]])
    # y_train = np.array([events_names[key] for key in y_train])
    # y_test = np.array([events_names[key] for key in y_test])
    # print(y_train[[0, 1, 2, 3, 4, -2, -1]])

    y_train = replace_label(y_train, '0', 'no_failure')
    y_test = replace_label(y_test, '0', 'no_failure')

    # print('unique ids', len(pd.concat([df_train_combined, df_test_combined])['id'].value_counts()))
    # print('number labels combined', str(len(y_test) + len(y_train)))
    # print('x_train', x_train.shape)
    # print('x_test', x_test.shape)
    # print('y_train', y_train.shape)
    # print('y_test', y_test.shape)

    if split_variant == SplitVariant.PREDEFINED:
        pass

    elif split_variant == SplitVariant.RANDOM_SPLIT:
        # combine into single df / list and split into train/test again
        # because the predefined split only has one class in test
        examples_array = np.concatenate([x_train, x_test], axis=0)
        labels_array = np.concatenate([y_train, y_test], axis=0)

        print('\nExecute train/test split')
        x_train, x_test, y_train, y_test = train_test_split(examples_array, labels_array,
                                                            test_size=(1 - split_range),
                                                            random_state=split_random_seed)
    elif split_variant == SplitVariant.DATE_BASED:
        date_threshold_typed = pd.to_datetime(date_threshold)

        x_train_new = []
        x_test_new = []
        y_train_new = []
        y_test_new = []
        dates_train_new = []
        dates_test_new = []

        examples_array = np.concatenate([x_train, x_test], axis=0)
        labels_array = np.concatenate([y_train, y_test], axis=0)
        dates = np.concatenate([dates_train, dates_test], axis=0)

        # go through each example and add it to the corresponding list based on its date compared to the threshold
        for x, y, date in zip(examples_array, labels_array, dates):
            date_typed = pd.to_datetime(date)

            if date_typed <= date_threshold_typed:
                x_train_new.append(x)
                y_train_new.append(y)
                dates_train_new.append(date)

            else:
                x_test_new.append(x)
                y_test_new.append(y)
                dates_test_new.append(date)

        x_train_new, x_test_new = np.array(x_train_new), np.array(x_test_new),
        y_train_new, y_test_new = np.array(y_train_new), np.array(y_test_new)
        dates_train_new, dates_test_new = np.array(dates_train_new), np.array(dates_test_new)

        x_train, x_test, y_train, y_test = x_train_new, x_test_new, y_train_new, y_test_new
        dates_train, dates_test = dates_train_new, dates_test_new

    elif split_variant == SplitVariant.ENSURE_NO_MIX:
        examples_array = np.concatenate([x_train, x_test], axis=0)
        labels_array = np.concatenate([y_train, y_test], axis=0)
        dates = np.concatenate([dates_train, dates_test], axis=0)

        gss = GroupShuffleSplit(n_splits=1, test_size=(1 - split_range), random_state=split_random_seed)

        train_indices, test_indices = list(gss.split(examples_array, labels_array, dates))[0]

        x_train, y_train, dates_train = examples_array[train_indices], labels_array[train_indices], dates[train_indices]
        x_test, y_test, dates_test = examples_array[test_indices], labels_array[test_indices], dates[test_indices]

        assert len(set(dates_train).intersection(set(dates_test))) == 0, 'Error: One date in both sets!'

    # reduce data arrays and column vector to sensor data columns only
    attribute_indices = [2, 3, 4, 5, 6, 7]
    attribute_names = attribute_names[attribute_indices]

    x_train = x_train[:, :, attribute_indices]
    x_test = x_test[:, :, attribute_indices]

    # normalize like for our dataset
    scaler_storage_path = str(working_directory) + '/scaler/'
    x_train, x_test = normalise(x_train, x_test, scaler_storage_path)

    # cast to float32 so it can directly be used by tensorflow
    x_train, x_test, = x_train.astype('float32'), x_test.astype('float32')

    print('\nOverview:')
    print('Train dataset shape:', x_train.shape)
    print('Train labels shape:', y_train.shape)
    print('Test dataset shape:', x_test.shape)
    print('Test labels shape:', y_test.shape)
    print()

    training_data_location = str(working_directory) + '/training_data/'
    print('\nExporting to: ', training_data_location)
    np.save(training_data_location + 'train_features.npy', x_train)
    np.save(training_data_location + 'test_features.npy', x_test)
    np.save(training_data_location + 'train_labels.npy', y_train)
    np.save(training_data_location + 'test_labels.npy', y_test)
    np.save(training_data_location + 'feature_names.npy', attribute_names)


# Nearly identical to the method in DatasetCreation.py, only path for storing the scalers was changed.
def normalise(x_train: np.ndarray, x_test: np.ndarray, path):
    length = x_train.shape[2]

    print('\nExecuting normalisation...')
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
        scaler_filename = path + 'scaler_' + str(i) + '.save'
        joblib.dump(scaler, scaler_filename)

    return x_train, x_test


####################################################################################################################

# Unchanged method from demo 2
def extract_samples(df, class_code):
    # Gets the observations labels and their unequivocal set
    ols = list(df['class'])
    set_ols = set()
    for ol in ols:
        if ol in set_ols or np.isnan(ol):
            continue
        set_ols.add(int(ol))

        # Discards the observations labels and replaces all nan with 0
    # (tsfresh's requirement)
    df_vars = df.drop('class', axis=1).fillna(0)

    # Initializes objects that will be return
    df_samples_train = pd.DataFrame()
    df_samples_test = pd.DataFrame()
    y_train = []
    y_test = []

    # Find out max numbers of samples from normal, transient and in regime periods
    #
    # Gets indexes (first and last) without overlap with other periods
    f_idx = ols.index(normal_class_code)
    l_idx = len(ols) - 1 - ols[::-1].index(normal_class_code)

    # Defines the initial numbers of samples for normal period
    max_samples_normal = l_idx - f_idx + 1 - sample_size
    if (max_samples_normal) > 0:
        num_normal_samples = min(max_samples_per_period, max_samples_normal)
        num_train_samples = int(split_range * num_normal_samples)
        num_test_samples = num_normal_samples - num_train_samples
    else:
        num_train_samples = 0
        num_test_samples = 0

    # Defines the max number of samples for transient period
    transient_code = class_code + 100
    if transient_code in set_ols:
        # Gets indexes (first and last) with possible overlap at the beginning
        # of this period
        f_idx = ols.index(transient_code)
        if f_idx - (sample_size - 1) > 0:
            f_idx = f_idx - (sample_size - 1)
        else:
            f_idx = 0
        l_idx = len(ols) - 1 - ols[::-1].index(transient_code)
        max_transient_samples = l_idx - f_idx + 1 - sample_size
    else:
        max_transient_samples = 0

        # Defines the max number of samples for in regime period
    if class_code in set_ols:
        # Gets indexes (first and last) with possible overlap at the beginning
        # or end of this period
        f_idx = ols.index(class_code)
        if f_idx - (sample_size - 1) > 0:
            f_idx = f_idx - (sample_size - 1)
        else:
            f_idx = 0
        l_idx = len(ols) - 1 - ols[::-1].index(class_code)
        if l_idx + (sample_size - 1) < len(ols) - 1:
            l_idx = l_idx + (sample_size - 1)
        else:
            l_idx = len(ols) - 1
        max_in_regime_samples = l_idx - f_idx + 1 - sample_size
    else:
        max_in_regime_samples = 0

        # Find out proper numbers of samples from normal, transient and in regime periods
    #
    num_transient_samples = ceil(num_test_samples / 2)
    num_in_regime_samples = num_test_samples - num_transient_samples
    if (max_transient_samples >= num_transient_samples) and \
            (max_in_regime_samples < num_in_regime_samples):
        num_in_regime_samples = max_in_regime_samples
        num_transient_samples = min(num_test_samples - num_in_regime_samples, max_transient_samples)
    elif (max_transient_samples < num_transient_samples) and \
            (max_in_regime_samples >= num_in_regime_samples):
        num_transient_samples = max_transient_samples
        num_in_regime_samples = min(num_test_samples - num_transient_samples, max_in_regime_samples)
    elif (max_transient_samples < num_transient_samples) and \
            (max_in_regime_samples < num_in_regime_samples):
        num_transient_samples = max_transient_samples
        num_in_regime_samples = max_in_regime_samples
        num_test_samples = num_transient_samples + num_in_regime_samples
    # print('num_train_samples: {}'.format(num_train_samples))
    # print('num_test_samples: {}'.format(num_test_samples))
    # print('num_transient_samples: {}'.format(num_transient_samples))
    # print('num_in_regime_samples: {}'.format(num_in_regime_samples))

    # Extracts samples from the normal period for training and for testing
    #
    # Gets indexes (first and last) without overlap with other periods
    f_idx = ols.index(normal_class_code)
    l_idx = len(ols) - 1 - ols[::-1].index(normal_class_code)

    # Defines the proper step and extracts samples
    if (num_normal_samples) > 0:
        if num_normal_samples == max_samples_normal:
            step_max = 1
        else:
            step_max = (max_samples_normal - 1) // (max_samples_per_period - 1)
        step_wanted = sample_size
        step = min(step_wanted, step_max)

        # Extracts samples for training
        sample_id = 0
        for idx in range(num_train_samples):
            f_idx_c = l_idx - sample_size + 1 - (num_normal_samples - 1 - idx) * step
            l_idx_c = f_idx_c + sample_size
            # print('{}-{}-{}'.format(idx, f_idx_c, l_idx_c))
            df_sample = df_vars.iloc[f_idx_c:l_idx_c, :]
            df_sample.insert(loc=0, column='id', value=sample_id)
            df_samples_train = df_samples_train.append(df_sample)
            y_train.append(normal_class_code)
            sample_id += 1

        # Extracts samples for testing
        sample_id = 0
        for idx in range(num_train_samples, num_train_samples + num_test_samples):
            f_idx_c = l_idx - sample_size + 1 - (num_normal_samples - 1 - idx) * step
            l_idx_c = f_idx_c + sample_size
            # print('{}-{}-{}'.format(idx, f_idx_c, l_idx_c))
            df_sample = df_vars.iloc[f_idx_c:l_idx_c, :]
            df_sample.insert(loc=0, column='id', value=sample_id)
            df_samples_test = df_samples_test.append(df_sample)
            y_test.append(normal_class_code)
            sample_id += 1

    # Extracts samples from the transient period (if it exists) for testing
    #
    if (num_transient_samples) > 0:
        # Defines the proper step and extracts samples
        if num_transient_samples == max_transient_samples:
            step_max = 1
        else:
            step_max = (max_transient_samples - 1) // (max_samples_per_period - 1)
        step_wanted = np.inf
        step = min(step_wanted, step_max)

        # Gets indexes (first and last) with possible overlap at the beginning of this period
        f_idx = ols.index(transient_code)
        if f_idx - (sample_size - 1) > 0:
            f_idx = f_idx - (sample_size - 1)
        else:
            f_idx = 0
        l_idx = len(ols) - 1 - ols[::-1].index(transient_code)

        # Extracts samples
        for idx in range(num_transient_samples):
            f_idx_c = f_idx + idx * step
            l_idx_c = f_idx_c + sample_size
            # print('{}-{}-{}'.format(idx, f_idx_c, l_idx_c))
            df_sample = df_vars.iloc[f_idx_c:l_idx_c, :]
            df_sample.insert(loc=0, column='id', value=sample_id)
            df_samples_test = df_samples_test.append(df_sample)
            y_test.append(transient_code)
            sample_id += 1

    # Extracts samples from the in regime period (if it exists) for testing
    #
    if (num_in_regime_samples) > 0:
        # Defines the proper step and extracts samples
        if num_in_regime_samples == max_in_regime_samples:
            step_max = 1
        else:
            step_max = (max_in_regime_samples - 1) // (max_samples_per_period - 1)
        step_wanted = sample_size
        step = min(step_wanted, step_max)

        # Gets indexes (first and last) with possible overlap at the beginning or end of this period
        f_idx = ols.index(class_code)
        if f_idx - (sample_size - 1) > 0:
            f_idx = f_idx - (sample_size - 1)
        else:
            f_idx = 0
        l_idx = len(ols) - 1 - ols[::-1].index(class_code)
        if l_idx + (sample_size - 1) < len(ols) - 1:
            l_idx = l_idx + (sample_size - 1)
        else:
            l_idx = len(ols) - 1

        # Extracts samples
        for idx in range(num_in_regime_samples):
            f_idx_c = f_idx + idx * step
            l_idx_c = f_idx_c + sample_size
            # print('{}-{}-{}'.format(idx, f_idx_c, l_idx_c))
            df_sample = df_vars.iloc[f_idx_c:l_idx_c, :]
            df_sample.insert(loc=0, column='id', value=sample_id)
            df_samples_test = df_samples_test.append(df_sample)
            y_test.append(class_code)
            sample_id += 1

    return df_samples_train, y_train, df_samples_test, y_test


# Unchanged method from demo 2
def class_and_file_generator(data_path, real=False, simulated=False, drawn=False):
    for class_path in data_path.iterdir():
        if class_path.is_dir():
            class_code = int(class_path.stem)
            for instance_path in class_path.iterdir():
                if (instance_path.suffix == '.csv'):
                    if (simulated and instance_path.stem.startswith('SIMULATED')) or \
                            (drawn and instance_path.stem.startswith('DRAWN')) or \
                            (real and (not instance_path.stem.startswith('SIMULATED')) and \
                             (not instance_path.stem.startswith('DRAWN'))):
                        yield class_code, instance_path


# Unchanged method from demo 2
def load_instance(instance_path):
    try:
        well, instance_id = instance_path.stem.split('_')
        df = pd.read_csv(instance_path, sep=',', header=0)
        assert (df.columns == columns).all(), 'invalid columns in the file {}: {}' \
            .format(str(instance_path), str(df.columns.tolist()))
        return df
    except Exception as e:
        raise Exception('error reading file {}: {}'.format(instance_path, e))


if __name__ == '__main__':
    main()
