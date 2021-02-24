import json

import pandas as pd

from configuration.Enums import BatchSubsetType, LossFunction, BaselineAlgorithm, SimpleSimilarityMeasure, \
    ArchitectureVariant, ComplexSimilarityMeasure, TrainTestSplitMode, AdjacencyMatrixPreprossingCNN2DWithAddInput,\
    NodeFeaturesForGraphVariants, AdjacencyMatrixType, FtDataSetVersion


####
# Note: Division into different classes only serves to improve clarity.
# Only the Configuration class should be used to access all variables.
# Important: It must be ensured that variable names are only used once.
# Otherwise they will be overwritten depending on the order of inheritance!
# All methods should be added to the Configuration class to be able to access all variables
####

class BaselineConfiguration:

    def __init__(self):
        ##
        # General
        ##

        # Output interval of how many examples have been compared so far. < 0 for no output
        self.baseline_temp_output_interval = -1

        self.baseline_use_relevant_only = False
        self.baseline_algorithm = BaselineAlgorithm.FEATURE_BASED_ROCKET

        # Should the case base representation be based on a fitting to the full training dataset?
        # True = Fit representation using full training dataset / Default: False = Fit representation on case base only.
        self.force_fit_on_full_training_data = False

        ##
        # Rocket
        ##

        self.rocket_kernels = 10_000  # 10_000 is the rocket default
        self.rocket_random_seed = 2342

        ###
        # Learning global similarity for baseline input
        ###
        # Tested configuration: True and standard_ffnn  with Rocket1NeuronDummy should be selected
        self.overwrite_input_data_with_baseline_representation = False  # default False


class GeneralConfiguration:

    def __init__(self):
        ###
        # This configuration contains overall settings that couldn't be match to a specific program component
        ###

        # Specifies the maximum number of gpus used
        self.max_gpus_used = 4

        # Specifies the maximum number of cores to be used
        self.max_parallel_cores = 30

        # Path and file name to the specific model that should be used for testing and live classification
        # Folder where the models are stored is prepended below
        self.filename_model_to_use = 'temp_snn_model_02-05_01-47-05_epoch-730'

        ##
        # Debugging - Don't use for feature implementation
        ##

        # Limit the groups that should be used for a cbs model
        # List content must match the group ids in config.json
        # Use = None or = [] for no restriction
        self.cbs_groups_used = []  # ['g0','g2', 'g3', 'g4', 'g5', 'g6', 'g7']

        # Select whether the group handlers of a cbs will will be executed in batches, so a single gpu is only used
        # by a single group handler during training and inference. Enabling this can be used if the models are to big
        # which would result in out of memory errors.
        # If using small models (which dont have oom problems) it is recommended to disable this function,
        # since this should result in a performance improvements
        self.batch_wise_handler_execution = False  # Default: False

        # Select the FT data set version: 2021 considers a holdout set (valid) whereas 2020 only has train and test sets
        self.ft_data_set_version = FtDataSetVersion.FT_DataSet_2021

class ModelConfiguration:

    def __init__(self):
        pass

        ###
        # This configuration contains all parameters defining the structure of the classifier.
        # (SNNs as well as the CBS similarity measure)
        ###

        ##
        # Architecture
        ##

        # Selection which basic architecture is used, see enum class for details
        self.architecture_variant = ArchitectureVariant.STANDARD_SIMPLE

        ##
        # Determines how the similarity between two embedding vectors is determined (when a simple architecture is used)
        ##

        # Most related work on time series with SNN use a fc layer at the end of a cnn to merge 1d-conv
        # features of time steps. Can be used via adding "fc_after_cnn1d_layers" in the hyperparameter configs file

        # Attention: Implementation expects a simple measure to return a similarity in the interval of [0,1]!
        # Only use euclidean_dis for TRAINING with contrastive loss
        self.simple_measure = SimpleSimilarityMeasure.ABS_MEAN

        self.complex_measure = ComplexSimilarityMeasure.BASELINE_OVERWRITE

        ###
        # Hyperparameters
        ###

        # Main directory where the hyperparameter config files are stored
        self.hyper_file_folder = '../configuration/hyperparameter_combinations/'
        self.use_hyper_file = True

        # If enabled each case handler of a CBS will use individual hyperparameters
        # No effect on SNN architecture
        self.use_individual_hyperparameters = False

        # If !use_individual_hyperparameters interpreted as a single json file, else as a folder
        # which contains json files named after the cases they should be used for
        # If no file with this name is present the 'default.json' Config will be used
        self.hyper_file = self.hyper_file_folder + 'cnn2d_withAddInput_GraphMasking\Proposed_Model_CNN-2D_GCN_Masked.json'

        ##
        # Various settings influencing the similarity calculation
        ##

        # SNN output is normalized (x = x/|x|) (useful for eucl.?)
        self.normalize_snn_encoder_output = False  # default: False

        # Additional option for encoder variant cnn2dwithaddinput and the euclidean distance:
        # Weighted euclidean similarity based on relevant attributes
        self.useFeatureWeightedSimilarity = True  # default: False

        # Weights are based on masking vectors that contain 1 if a feature is selected as relevant for a
        # label (failure mode) and 0 otherwise. If option is set False then features based
        # on groups are used.

        # Select whether the reduction to relevant features should be based on the case itself or the group it belongs
        # to. Based on case = True, based on group = False
        # Must be false for CBS!
        self.individual_relevant_feature_selection = True  # default: True

        # Using the more restrictive features as additional masking vector for feature sim calculation
        # in cnn_with_add_input
        self.use_additional_strict_masking_for_attribute_sim = True  # default: False

        # Changes the masks to generate noise and provide overfitting (for test purposes)
        self.use_masking_regularization = False # default: False

        # Option to simulate a retrieval situation (during training) where only the weights of the
        # example from the case base/training data set are known:
        self.use_same_feature_weights_for_unsimilar_pairs = True   # default: True

        # Compares each time step of the encoded representation with each other time step
        # (instead of only comparing the ones with the same indices)
        # Implementation is based on NeuralWarp FFNN but used for simple similarity measures
        self.use_time_step_wise_simple_similarity = False  # default: False

        # Using additional static node features
        self.use_additional_static_node_features_for_graphNN = NodeFeaturesForGraphVariants.OWL2VEC_EMBEDDINGS_DIM16  # default: False


class TrainingConfiguration:

    def __init__(self):
        ###
        # This configuration contains all parameters defining the way the model is trained
        ###

        # Important: CBS will only function correctly if cbs_features or a superset of it is selected
        # cbs_features for SNNs will use the a subset of all_features, which are considered to be relevant
        # for at least one case
        # self.features_used will be assigned when config.json loading
        self.feature_variants = ['all_features', 'cbs_features']
        self.feature_variant = self.feature_variants[0]
        self.features_used = None

        # TODO Distance-Based Logistic Loss
        self.type_of_loss_function = LossFunction.BINARY_CROSS_ENTROPY

        # Settings for constrative_loss
        self.margin_of_loss_function = 2

        # Scalar margin h for triplet loss function
        self.triplet_loss_margin_h = 3

        # Reduce margin of constrative_loss or in case of binary cross entropy loss
        # smooth negative examples by half of the sim between different labels
        self.use_margin_reduction_based_on_label_sim = False  # default: False

        self.use_early_stopping = True
        self.early_stopping_epochs_limit = 500
        self.early_stopping_loss_minimum = -1  # Default: -1.0 (no effect), CNN2D_with_add_Input: BCE:0.03, MSE:0.01

        # Parameter to control if and when a test is conducted through training
        self.use_inference_test_during_training = False  # default False
        self.inference_during_training_epoch_interval = 10000

        # Definition of batch compositions
        # Key = Enum for selecting how the pairs are chosen, value = size of the subset of this type, must add up to 1.0
        # The same number of positive and negative pairs are generated for each type
        self.batch_distribution = {
            BatchSubsetType.DISTRIB_BASED_ON_DATASET: 0.5,
            BatchSubsetType.EQUAL_CLASS_DISTRIB: 0.5
        }

        # Use a custom similarity values instead of 0 for unequal / negative pairs during batch creation
        # These are based on the similarity matrices loaded in the dataset
        self.use_sim_value_for_neg_pair = False  # default: False

        # Defines how often loss is printed and checkpoints are saved during training
        self.output_interval = 1

        # How many model checkpoints are kept
        self.model_files_stored = 10001

        # Select which adjacency matrix is used
        self.adj_matrix_type = AdjacencyMatrixType.ADJ_MAT_TYPE_AS_ONE_GRAPH_SPARSE

        # Define how the adjacency matrix is computed for cnn2dwithAddInput
        self.use_predefined_adj_matrix_as_base_for_preprocessing = True
        self.use_GCN_adj_matrix_preprocessing = True
        self.use_GCN_adj_matrix_preprocessing_sym = False
        self.add_selfloop_to_adj_matrix = False
        self.adj_matrix_preprocessing = AdjacencyMatrixPreprossingCNN2DWithAddInput.ADJ_MATRIX_CONTEXT_GCN

class InferenceConfiguration:

    def __init__(self):
        ##
        # Settings and parameters for all inference processes
        ##
        # Notes:
        #   - Folder of used model is specified in GeneralConfiguration
        #   - Does not include settings for BaselineTester

        # If enabled only the reduced training dataset (via CaseBaseExtraction) will be used for
        # similarity assessment during inference.
        # Please note that the case base extraction only reduces the training data but fully copies the test data
        # so all test example will still be evaluated even if this is enabled
        self.case_base_for_inference = True  # default: False

        # Parameter to control the size / number of the queries used for evaluation
        self.inference_with_failures_only = False  # default: False

        # If enabled the similarity assessment of the test dataset to the training datset will be split into chunks
        # Possibly necessary due to VRam limitation
        self.split_sim_calculation = True  # default False
        self.sim_calculation_batch_size = 128

        # If enabled the model is printed as model.png
        self.print_model = False

        # In case of too small distance values (resulting in 1.0) through similarity transformation
        self.distance_scaling_parameter_for_cnn2dwithAddInput_ontopNN = 1.0

        # With TrainSelectAndTest is used, how many models (ascending from the best loss) should be tested?
        self.number_of_selection_tests = 6 #12#10#7


        # Use valid data set instead of test (e.g., during inference)
        self.use_valid_instead_of_test = False                                                           # default False


class ClassificationConfiguration:

    def __init__(self):
        ###
        # This configuration contains settings regarding the real time classification
        # and the therefore required Kafka server and case base
        ###
        # Note: Folder of used model specified in GeneralConfiguration

        # server information
        self.ip = 'localhost'  # '192.168.1.10'
        self.port = '9092'

        self.error_descriptions = None  # Read from config.json

        # Set to true if using the fabric simulation (FabricSimulation.py)
        # This setting causes the live classification to read from the beginning of the topics on the Kafka server,
        # so the simulation only has to be run only once.
        self.testing_using_fabric_sim = True

        # Enables the functionality to export the classification results back to the Kafka server
        self.export_results_to_kafka = True

        # Topic where the messages should be written to. Automatically created if not existing.
        self.export_topic = 'classification-results'

        # Determines on which topic's messages the time interval for creating an example is based on
        # Only txt topics can be used
        self.limiting_topic = 'txt15'

        ###
        # Case base
        ###

        # the random seed the index selection is based on
        self.random_seed_index_selection = 42

        # the number of examples per class the training data set should be reduced to for the live classification
        self.examples_per_class = 150  # default: 150

        # the k of the knn classifier used for live classification
        self.k_of_knn = 10


class PreprocessingConfiguration:

    def __init__(self):
        ###
        # This configuration contains information and settings relevant for the data preprocessing and dataset creation
        ###

        ##
        # Import and data visualisation
        ##

        self.plot_txts: bool = False
        self.plot_pressure_sensors: bool = False
        self.plot_acc_sensors: bool = False
        self.plot_bmx_sensors: bool = False
        self.plot_all_sensors: bool = False

        self.export_plots: bool = True

        self.print_column_names: bool = False
        self.save_pkl_file: bool = True

        ##
        # Preprocessing
        ##

        # Value is used to ensure a constant frequency of the measurement time points
        self.resample_frequency = "4ms"  # need to be the same for DataImport as well as DatasetCreation

        # Define the length (= the number of timestamps) of the time series generated
        self.time_series_length = 1000

        # Defines the step size of window, e.g. for = 2: Example 0 starts at 00:00:00 and Example 1 at 00:00:02
        # For no overlapping: value = seconds(time_series_length * resample frequency)
        self.overlapping_window_step_seconds = 1

        # Configure the motor failure parameters used in case extraction
        self.split_t1_high_low = True
        self.type1_start_percentage = 0.5
        self.type1_high_wear_rul = 25
        self.type2_start_rul = 25

        # seed for how the train/test data is split randomly
        self.random_seed = 41

        # share of examples used as test set
        self.test_split_size = 0.2

        # way examples are split into train and test, see enum class for details
        self.split_mode = TrainTestSplitMode.ANOMALY_DETECTION

        ##
        # Lists of topics separated by types that need different import variants
        ##

        self.txt_topics = ['txt15', 'txt16', 'txt17', 'txt18', 'txt19']

        # Unused topics: 'bmx055-VSG-gyr','bmx055-VSG-mag','bmx055-HRS-gyr','bmx055-HRS-mag'
        self.acc_topics = ['adxl0', 'adxl1', 'adxl2', 'adxl3']

        self.bmx_acc_topics = []  # unused topics: 'bmx055-VSG-acc', 'bmx055-HRS-acc'

        self.pressure_topics = ['pressureSensors']

        self.pressure_sensor_names = ['Oven', 'VSG']  # 'Sorter' not used

        # Combination of all topics in a single list
        self.topic_list = self.txt_topics + self.acc_topics + self.bmx_acc_topics + self.pressure_topics


class StaticConfiguration:

    def __init__(self, dataset_to_import):
        ###
        # This configuration contains data that rarely needs to be changed, such as the paths to certain directories
        ###

        ##
        # Static values
        ##
        # All of the following None-Variables are read from the config.json file because they are mostly static
        # and don't have to be changed very often

        self.cases_datasets, self.datasets = None, None

        # mapping for topic name to prefix of sensor streams, relevant to get the same order of streams
        self.prefixes = None

        self.case_to_individual_features = None
        self.case_to_individual_features_strict = None
        self.case_to_group_id = None
        self.group_id_to_cases = None
        self.group_id_to_features = None

        self.type_based_groups = {}

        self.zeroOne, self.intNumbers, self.realValues, self.categoricalValues = None, None, None, None

        # noinspection PyUnresolvedReferences
        self.load_config_json('../configuration/config.json')

        ##
        # Folders and file names
        ##
        # Note: Folder of used model specified in GeneralConfiguration

        # Default prefix for main dataset
        self.data_folder_prefix = '../data/'
        # Prefix for the 3w dataset
        # self.data_folder_prefix = '../data/additional_datasets/3w_dataset/'
        # self.data_folder_prefix = 'D:/Seafile/Seafile/PredMDataSet2021/'

        # Folder where the trained models are saved to during learning process
        self.models_folder = self.data_folder_prefix #+ 'trained_models/'

        # noinspection PyUnresolvedReferences
        self.directory_model_to_use = self.models_folder + self.filename_model_to_use + '/'

        # Folder where the preprocessed training and test data for the neural network should be stored
        self.training_data_folder = self.data_folder_prefix + 'training_data/' # 'case_base/' for training tsfresh

        # Folder where the normalisation models should be stored
        self.scaler_folder = self.data_folder_prefix + 'scaler/'

        # Name of the files the dataframes are saved to after the import and cleaning
        self.filename_pkl = 'export_data.pkl'
        self.filename_pkl_cleaned = 'cleaned_data.pkl'

        # Folder where the reduced training data set aka. case base is saved to
        self.case_base_folder = self.data_folder_prefix + 'case_base/'

        # Folder where text files with extracted cases are saved to, for export
        self.cases_folder = self.data_folder_prefix + 'cases/'

        # File from which the case information should be loaded, used in dataset creation
        self.case_file = '../configuration/cases.csv'

        # Custom measure sim matrices
        self.condition_sim_matrix_file = self.training_data_folder + 'Condition_Sim_Matrix.csv'
        self.failure_mode_sim_matrix_file = self.training_data_folder + 'FailureMode_Sim_Matrix.csv'
        self.localisation_sim_matrix_file = self.training_data_folder + 'Localization_Sim_Matrix.csv'

        # CSV file containing the adjacency information of features used by the graph cnn2d encoder
        if self.adj_matrix_type == AdjacencyMatrixType.ADJ_MAT_TYPE_AS_ONE_GRAPH_SPARSE:
            self.graph_adjacency_matrix_attributes_file = self.training_data_folder + 'adjacency_matrix_v3_fullGraph_sparse.CSV'
        elif self.adj_matrix_type == AdjacencyMatrixType.ADJ_MAT_TYPE_AS_ONE_GRAPH_WS_FULLY:
            self.graph_adjacency_matrix_attributes_file = self.training_data_folder + 'adjacency_matrix_v3_fullGraph_wsFullyConnected.CSV'
        elif self.adj_matrix_type == AdjacencyMatrixType.ADJ_MAT_TYPE_FIRST_VARIANT:
            self.graph_adjacency_matrix_attributes_file = self.training_data_folder + 'adjacency_matrix.CSV'
        elif self.adj_matrix_type == AdjacencyMatrixType.ADJ_MAT_TYPE_FULLY_CONNECTED:
            self.graph_adjacency_matrix_attributes_file = self.training_data_folder + 'adjacency_matrix_all_attributes_allOne.csv'
        else:
            raise AttributeError('Unknown adj. matrix variant:', self.adj_matrix_type)
        #self.graph_adjacency_matrix_attributes_file = self.training_data_folder + 'adjacency_matrix_v3_fullGraph_sparse.CSV' #'adjacency_matrix.CSV'#'adjacency_matrix_v3_fullGraph_sparse.CSV' #'adjacency_matrix_all_attributes_allOne.csv'
        self.graph_adjacency_matrix_ws_file = self.training_data_folder + 'adjacency_matrix_wokstation.csv'
        self.graph_attr_to_workstation_relation_file = self.training_data_folder + 'attribute_to_txtcontroller.csv'
        self.mapping_attr_to_ftonto_file = self.training_data_folder + 'mapping_attr_to_ftonto-uri.csv'
        if self.use_additional_static_node_features_for_graphNN == NodeFeaturesForGraphVariants.OWL2VEC_EMBEDDINGS_DIM16:
            self.graph_owl2vec_node_embeddings_file = self.training_data_folder + 'owl2vec_node_embeddings_dim16.csv'
        elif self.use_additional_static_node_features_for_graphNN == NodeFeaturesForGraphVariants.OWL2VEC_EMBEDDINGS_DIM32:
            self.graph_owl2vec_node_embeddings_file = self.training_data_folder + 'owl2vec_node_embeddings_dim32.csv'

        # TS Fresh feature files
        self.ts_fresh_filtered_file = 'ts_fresh_extracted_features_filtered.pkl'
        self.ts_fresh_unfiltered_file = 'ts_fresh_extracted_features_unfiltered.pkl'
        self.ts_fresh_filtered_file_scaled = "ts_fresh_extracted_features_filtered_min_max_scaled.npy"
        self.ts_fresh_unfiltered_file_scaled = "ts_fresh_extracted_features_unfiltered_min_max_scaled.npy"

        # Rocket feature files
        self.rocket_features_train_file = 'rocket_features_train_new2.npy'
        self.rocket_features_test_file = 'rocket_features_test_new2.npy'
        self.rocket_features_valid_file = 'rocket_features_valid_new2.npy'

        # Select specific dataset with given parameter
        # Preprocessing however will include all defined datasets
        self.pathPrefix = self.datasets[dataset_to_import][0]
        self.startTimestamp = self.datasets[dataset_to_import][1]
        self.endTimestamp = self.datasets[dataset_to_import][2]

        # Query to reduce datasets to the given time interval
        self.query = "timestamp <= \'" + self.endTimestamp + "\' & timestamp >= \'" + self.startTimestamp + "\' "

        # Define file names for all topics
        self.txt15 = self.pathPrefix + 'raw_data/txt15.txt'
        self.txt16 = self.pathPrefix + 'raw_data/txt16.txt'
        self.txt17 = self.pathPrefix + 'raw_data/txt17.txt'
        self.txt18 = self.pathPrefix + 'raw_data/txt18.txt'
        self.txt19 = self.pathPrefix + 'raw_data/txt19.txt'

        self.topicPressureSensorsFile = self.pathPrefix + 'raw_data/pressureSensors.txt'

        self.acc_txt15_m1 = self.pathPrefix + 'raw_data/TXT15_m1_acc.txt'
        self.acc_txt15_comp = self.pathPrefix + 'raw_data/TXT15_o8Compressor_acc.txt'
        self.acc_txt16_m3 = self.pathPrefix + 'raw_data/TXT16_m3_acc.txt'
        self.acc_txt18_m1 = self.pathPrefix + 'raw_data/TXT18_m1_acc.txt'

        self.bmx055_HRS_acc = self.pathPrefix + 'raw_data/bmx055-HRS-acc.txt'
        self.bmx055_HRS_gyr = self.pathPrefix + 'raw_data/bmx055-HRS-gyr.txt'
        self.bmx055_HRS_mag = self.pathPrefix + 'raw_data/bmx055-HRS-mag.txt'

        self.bmx055_VSG_acc = self.pathPrefix + 'raw_data/bmx055-VSG-acc.txt'
        self.bmx055_VSG_gyr = self.pathPrefix + 'raw_data/bmx055-VSG-gyr.txt'
        self.bmx055_VSG_mag = self.pathPrefix + 'raw_data/bmx055-VSG-mag.txt'


class Configuration(
    PreprocessingConfiguration,
    ClassificationConfiguration,
    InferenceConfiguration,
    TrainingConfiguration,
    ModelConfiguration,
    GeneralConfiguration,
    StaticConfiguration,
    BaselineConfiguration
):

    def __init__(self, dataset_to_import=0):
        PreprocessingConfiguration.__init__(self)
        ClassificationConfiguration.__init__(self)
        InferenceConfiguration.__init__(self)
        TrainingConfiguration.__init__(self)
        ModelConfiguration.__init__(self)
        GeneralConfiguration.__init__(self)
        StaticConfiguration.__init__(self, dataset_to_import)
        BaselineConfiguration.__init__(self)

    def load_config_json(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)

        self.datasets = data['datasets']
        self.prefixes = data['prefixes']
        self.error_descriptions = data['error_descriptions']
        self.zeroOne = data['zeroOne']
        self.intNumbers = data['intNumbers']
        self.realValues = data['realValues']
        self.categoricalValues = data['categoricalValues']

        self.label_renaming_overall = data['label_renaming_overall']
        self.label_renaming = data['label_renaming']
        self.transfer_from_train_to_test = data['transfer_from_train_to_test']
        self.unused_labels = data['unused_labels']

        # Remove duplicates to ensure output is correct (would result in wrong sum of changed examples otherwise)
        self.unused_labels = list(set(self.unused_labels))
        self.transfer_from_train_to_test = list(set(self.transfer_from_train_to_test))

        def flatten(list_of_lists):
            return [item for sublist in list_of_lists for item in sublist]

        self.case_to_individual_features = data['relevant_features']
        if self.use_additional_strict_masking_for_attribute_sim:
            self.case_to_individual_features_strict = data['relevant_features_strict']
        self.case_to_group_id: dict = data['case_to_group_id']
        self.group_id_to_cases: dict = data['group_id_to_cases']
        self.group_id_to_features: dict = data['group_id_to_features']

        # Keys must be converted to integer, json only uses strings
        type_based_groups_json = data['type_based_groups']
        for key, value in type_based_groups_json.items():
            self.type_based_groups[int(key)] = value

        # Depending on the self.feature_variant the relevant features for creating a dataset are selected
        if self.feature_variant == 'cbs_features':
            self.features_used = sorted(list(set(flatten(self.group_id_to_features.values()))))
        elif self.feature_variant == 'all_features':
            self.features_used = sorted(data['all_features'])
        else:
            raise AttributeError('Unknown feature variant:', self.feature_variant)

    def get_relevant_features_group(self, case):
        group = self.case_to_group_id.get(case)
        return self.group_id_to_features.get(group)

    # returns individual defined features (instead of group features)
    def get_relevant_features_case(self, case, return_strict_masking=False):
        if return_strict_masking:
            return [self.case_to_individual_features.get(case), self.case_to_individual_features_strict.get(case)]
        else:
            return self.case_to_individual_features.get(case)

    # return the error case description for the passed label
    def get_error_description(self, error_label: str):
        return self.error_descriptions[error_label]

    def get_connection(self):
        return self.ip + ':' + self.port

    # import the timestamps of each dataset and class from the cases.csv file
    def import_timestamps(self):
        datasets = []
        number_to_array = {}

        with open(self.case_file, 'r') as file:
            for line in file.readlines():
                parts = line.split(',')
                parts = [part.strip(' ') for part in parts]
                # print("parts: ", parts)
                # dataset, case, start, end = parts
                dataset = parts[0]
                case = parts[1]
                start = parts[2]
                end = parts[3]
                failure_time = parts[4].rstrip()

                timestamp = (gen_timestamp(case, start, end, failure_time))

                if dataset in number_to_array.keys():
                    number_to_array.get(dataset).append(timestamp)
                else:
                    ds = [timestamp]
                    number_to_array[dataset] = ds

        for key in number_to_array.keys():
            datasets.append(number_to_array.get(key))

        self.cases_datasets = datasets

    def print_detailed_config_used_for_training(self):
        print("--- Current Configuration ---")
        print("General related:")
        print("- simple_measure: ", self.simple_measure)
        print("- hyper_file: ", self.hyper_file)
        print("- self.models_folder: ",self.models_folder)
        print("")
        print("Masking related:")
        print("- individual_relevant_feature_selection: ", self.individual_relevant_feature_selection)
        print("- use_additional_strict_masking_for_attribute_sim: ",
              self.use_additional_strict_masking_for_attribute_sim)
        print("- use_same_feature_weights_for_unsimilar_pairs: ", self.use_same_feature_weights_for_unsimilar_pairs)
        print("- use_masking_regularization: ", self.use_masking_regularization)
        print("")
        print("Similarity Measure related:")
        print("- useFeatureWeightedSimilarity: ", self.useFeatureWeightedSimilarity)
        print("- use_time_step_wise_simple_similarity: ", self.use_time_step_wise_simple_similarity)
        print("- feature_variant: ", self.feature_variant)
        print("")
        print("Graph Input related:")
        print("- adj_matrix_type: ", self.adj_matrix_type)
        print("- use_predefined_adj_matrix_as_base_for_preprocessing: ", self.use_predefined_adj_matrix_as_base_for_preprocessing)
        print("- use_GCN_adj_matrix_preprocessing: ", self.use_GCN_adj_matrix_preprocessing)
        print("- use_GCN_adj_matrix_preprocessing_sym: ", self.use_GCN_adj_matrix_preprocessing_sym)
        print("- add_selfloop_to_adj_matrix: ", self.add_selfloop_to_adj_matrix)
        print("- adj_matrix_preprocessing: ", self.adj_matrix_preprocessing)

        print("Static Node Features: ")
        print("use_additional_static_node_features_for_graphNN: " ,self.use_additional_static_node_features_for_graphNN)
        print("")
        print("Training related:")
        print("- batch_distribution: ", self.batch_distribution)
        print("- type_of_loss_function: ", self.type_of_loss_function)
        print("- margin_of_constrative_loss_function: ", self.margin_of_loss_function)
        print("- margin_of_triplet_loss_function: ", self.triplet_loss_margin_h)
        print("- use_margin_reduction_based_on_label_sim: ", self.use_margin_reduction_based_on_label_sim)
        print("- use_sim_value_for_neg_pair: ", self.use_sim_value_for_neg_pair)
        print("- use_early_stopping: ", self.use_early_stopping)
        print("- early_stopping_epochs_limit: ", self.early_stopping_epochs_limit)
        print("- early_stopping_loss_minimum: ", self.early_stopping_loss_minimum)
        print("- model_files_stored: ", self.model_files_stored)
        print("- output_interval: ", self.output_interval)
        print("")
        print("Inference related:")
        print("- case_base_for_inference: ", self.case_base_for_inference)
        print("- split_sim_calculation: ", self.split_sim_calculation)
        print("- sim_calculation_batch_size: ", self.sim_calculation_batch_size)
        print("- number_of_selection_tests: ", self.number_of_selection_tests)
        print("--- ---")


def gen_timestamp(label: str, start: str, end: str, failure_time: str):
    start_as_time = pd.to_datetime(start, format='%Y-%m-%d %H:%M:%S.%f')
    end_as_time = pd.to_datetime(end, format='%Y-%m-%d %H:%M:%S.%f')
    if failure_time != "no_failure":
        failure_as_time = pd.to_datetime(failure_time, format='%Y-%m-%d %H:%M:%S')
    else:
        failure_as_time = ""

    # return tuple consisting of a label and timestamps in the pandas format
    return label, start_as_time, end_as_time, failure_as_time
