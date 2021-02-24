from enum import Enum


class BatchSubsetType(Enum):
    # The probability of the classes is equally distributed
    EQUAL_CLASS_DISTRIB = 0
    # The probability of the classes is based on the dataset (= number of examples with a class / number of all examples)
    DISTRIB_BASED_ON_DATASET = 1
    # Pairs only consist of examples of failure modes, probability like 1
    ONLY_FAILURE_PAIRS = 2
    # Positive pairs will be the same as for 2, but negative pairs could also be failure mode, no_failure
    NO_FAILURE_ONLY_FOR_NEG_PAIRS = 3


class LossFunction(Enum):
    BINARY_CROSS_ENTROPY = 0
    CONSTRATIVE_LOSS = 1
    MEAN_SQUARED_ERROR = 2
    HUBER_LOSS = 3
    TRIPLET_LOSS = 4


class BaselineAlgorithm(Enum):
    DTW = 0
    DTW_WEIGHTING_NBR_FEATURES = 1
    FEATURE_BASED_TS_FRESH = 2
    FEATURE_BASED_ROCKET = 3


class ArchitectureVariant(Enum):
    # standard = classic snn behaviour, context vectors calculated each time, also multiple times for the example
    # fast = encoding of case base only once, example also only once
    # simple = a simple similarity measure is used to compare the encoder outputs, configure in config.simple_measure
    # complex = a neural network is used to determine similarity between encodings, configured in config.complex_measure

    STANDARD_SIMPLE = 0
    FAST_SIMPLE = 1
    STANDARD_COMPLEX = 2
    FAST_COMPLEX = 3

    @staticmethod
    def is_simple(av):
        return av in [ArchitectureVariant.STANDARD_SIMPLE, ArchitectureVariant.FAST_SIMPLE]

    @staticmethod
    def is_complex(av):
        return not ArchitectureVariant.is_simple(av)

    @staticmethod
    def is_fast(av):
        return av in [ArchitectureVariant.FAST_SIMPLE, ArchitectureVariant.FAST_COMPLEX]


class SimpleSimilarityMeasure(Enum):
    ABS_MEAN = 0
    EUCLIDEAN_SIM = 1
    EUCLIDEAN_DIS = 2
    COSINE = 3


class ComplexSimilarityMeasure(Enum):
    FFNN_NW = 0
    GRAPH_SIM = 1
    BASELINE_OVERWRITE = 2
    CNN2DWAddInp = 3


class TrainTestSplitMode(Enum):
    # Examples of a single run to failure are not in both train and test
    ENSURE_NO_MIX = 0

    # Train only consists of no_failure examples, also includes ENSURE_NO_MIX guarantee
    ANOMALY_DETECTION = 1

class AdjacencyMatrixPreprossingCNN2DWithAddInput(Enum):
    # The type of adj matrix used
    ADJ_MATRIX_CONTEXT_GCN = 0
    ADJ_MATRIX_STRICT_GCN = 1
    ADJ_MATRIX_STRICT_WITH_CONTEXT_DIFF_GCN = 2
    ADJ_MATRIX_BLANK_FOR_FAST_COMPUTATION = 3

class NodeFeaturesForGraphVariants(Enum):
    # The type of node features used
    NO_ADDITIONAL_NODE_FEATURES = 0
    ONE_HOT_ENCODED = 1
    OWL2VEC_EMBEDDINGS_DIM16 = 2
    OWL2VEC_EMBEDDINGS_DIM32 = 3

class AdjacencyMatrixType(Enum):
    # The type of adj matrix used
    ADJ_MAT_TYPE_AS_ONE_GRAPH_SPARSE = 0
    ADJ_MAT_TYPE_AS_ONE_GRAPH_WS_FULLY = 1
    ADJ_MAT_TYPE_FIRST_VARIANT = 2
    ADJ_MAT_TYPE_FULLY_CONNECTED = 3

class FtDataSetVersion(Enum):
    # The type of Fischertechnik Factory Model Data Set is used
    FT_DataSet_2020 = 0
    FT_DataSet_2021 = 1
    FT_DataSet_2021_FewShotK3 = 2