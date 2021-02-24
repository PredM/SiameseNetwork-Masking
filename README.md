#  Using Expert Knowledge for Masking Irrelevant Data Streams in Siamese Networks for the Detection and Prediction of Faults
Accompanying repository to the paper [Using Expert Knowledge for Masking Irrelevant Data Streams in Siamese Networks  for the Detection and Prediction of Faults](https://PLACEHOLDER.com) which is currently under review.

The implementation of some components is based on the one presented in [NeuralWarp](https://arxiv.org/abs/1812.08306) ([GitHub](https://github.com/josifgrabocka/neuralwarp)).

## Supplementary Resources
* The sub directory [supplementary\_resources](https://github.com/PredM/SiameseNetwork-Masking/tree/master/supplementary_resources) of this repository contains additional information about the [data sets](https://github.com/PredM/SiameseNetwork-Masking/blob/master/supplementary_resources/Train_Valid_Test_DataSet_Statistics.png) used and the architecture of the CNN-2D GCN Masked model([Sketch](https://github.com/PredM/SiameseNetwork-Masking/blob/master/supplementary_resources/2DCNN-GCN_Masked_Architecture.png), [Plotted Model Graph](https://github.com/PredM/SiameseNetwork-Masking/blob/master/supplementary_resources/CNN2D-GCN_Masked.png)).
* The [detailed logs](https://seafile.rlp.net/d/229942a2015f46648e5c/) for each of the experiments presented in Tab. 2 and 3 are provided. 
* The [raw data](https://seafile.rlp.net/d/cd5590e4e9d249b2847e/) recorded with this [simulation factory model](https://iot.uni-trier.de) was used to generate the training and evaluation data sets.
* The [preprocessed data set](https://seafile.rlp.net/d/fcf7958432af4ecfb380/) we used for the evaluation.

## Quick start guide: How to train and evaluate a model?
1. Clone the repository
2. Download the [preprocessed data set](https://seafile.rlp.net/d/fcf7958432af4ecfb380/) and move it to the _data_ folder
3. Navigate to the _neural_network_ folder and start the training and test procedure via _python [TrainSelectAndTest.py](https://github.com/PredM/SiameseNetwork-Masking/blob/master/neural_network/TrainSelectAndTest.py) > Log.txt_. This will reproduce the results for the proposed Model ([CNN-2D GCN Masked](https://github.com/PredM/SiameseNetwork-Masking/blob/0ca2daea6092058c86ab34e48e1644dbb633a08c/neural_network/BasicNeuralNetworks.py#L536), c.f. Sect. 3)

You can train and test another SNN architecture by changing the encoder configuration in [line 130 in Configuration.py](https://github.com/PredM/SiameseNetwork-Masking/blob/0ca2daea6092058c86ab34e48e1644dbb633a08c/configuration/Configuration.py#L130).

## Requirements
Used python version: 3.7.X \
Used packages: See requirements.txt

## Used Hardware
<table>
    <tr>
        <td>CPU</td>
        <td>2x 40x Intel Xeon Gold 6138 @ 2.00GHz</td>
    </tr>
    <tr>
        <td>RAM</td>
        <td>12 x 64 GB Micron DDR4</td>
    </tr>
       <tr>
        <td>GPU</td>
        <td>8 x NVIDIA Tesla V100 32 GB GPUs</td>
    </tr>
</table>

## General instructions for use
* All settings can be adjusted in the script [Configuration.py](https://github.com/PredM/SiameseNetwork-Masking/blob/master/configuration/Configuration.py), 
whereby some rarely changed variables such as the [relevant data streams per class](https://github.com/PredM/SiameseNetwork-Masking/blob/0ca2daea6092058c86ab34e48e1644dbb633a08c/configuration/config.json#L314) are stored in the file [config.json](https://github.com/PredM/SiameseNetwork-Masking/blob/master/configuration/config.json), which is read in during the initialization.
* The hyperparameters of the neural networks can be defined in the script Hyperparameter.py or a pre-defined configuration can be imported from a file in [configuration/hyperparameter_combinations/](https://github.com/PredM/SiameseNetwork-Masking/tree/master/configuration/hyperparameter_combinations).
* For training, selecting the best model on the validation set and testing it on the test set, use:[TrainSelectAndTest.py](https://github.com/PredM/SiameseNetwork-Masking/blob/master/neural_network/TrainSelectAndTest.py) 
* The [data/ directory](https://github.com/PredM/SiameseNetwork-Masking/tree/master/data) contains all required data. Central are the pre-processed training data in [data/training_data/](https://github.com/PredM/SiameseNetwork-Masking/tree/master/data/training_data9 and the trained models in [data/trained_models/](https://github.com/PredM/SiameseNetwork-Masking/tree/master/data/trained_models). 
A detailed description of what each directory contains is given in corresponding parts of the configuration file. 

## Software components
The following section gives an overview of the most important packages, directories and included Python scripts in this repository. 

### configuration
| Python script | Purpose |
| ---      		|  ------  |
|Configuration.py|The configuration file within which all adjustments can be made.|
|Hyperparameters.py| Contains the class that stores the hyperparameters used by a single neural network.|

### neural_network
| Python script | Purpose |
| ---      		|  ------  |
|BasicNeuralNetworks.py| Contains the implementation of all basic types of neural networks, e.g. CNN, FFNN.|
|Dataset.py|Contains the class that stores the training data and meta data about it. Used by any scripts that uses the generated dataset|
|Evaluator.py|Contains an evaluation procedure which is used by all test routines, i.e. SNNs, CBS and baseline testers.|
|Inference.py|Provides the ability to test a trained model on the test data set.|
|Optimizer.py|Contains the optimizer routine for updating the parameters during training. Used for optimizing SNNs as well as the CBS.|
|SimpleSimilarityMeasure.py|Several simple similarity measures for calculating the similarity between the enbedding vectors are implemented here.|
|SNN.py|Includes all four variants of the siamese neural network (classic architecture or optimized variant, simple or FFNN similiarty measure).|
|TrainAndTest.py| Execution of a training followed by automatic evaluation of the model with best loss.|
|TrainSelectAndTest.py| Execution of a training followed by automatic selection of the models with best results on the validation set and finally evaluation on the test set.|
|Training.py| Used to execute the training process.|

### baseline
| Python script | Purpose |
| ---      		|  ------  |
|BaselineTester.py| Provides the possibility to apply other methods for determining similarities of time series, e.g. DTW, tsfresh, ROCKET to the data set. |

### data_processing
| Python script | Purpose |
| ---      		|  ------  |
|CaseBaseExtraction.py| Provides extraction of a case base from the entire training data set.|
|DataImport.py|This script executes the first part of the preprocessing. It consists of reading the unprocessed sensor data from Kafka topics in JSON format as a *.txt file (e.g., acceleration, BMX, txt, print) and then saving it as export_data.pkl in the same folder. This script also defines which attributes/features/streams are used via config.json with the entry "relevant_features". Which data is processed can also be set in config.json with the entry datasets (path, start, and end timestamp). |
|DataframeCleaning.py|This script executes the second part of the preprocessing of the training data. It needs the export_data.pkl file generated in the first step. The cleanup procedure consists of the following steps: 1. Replace True/False with 1/0, 2. Fill NA for boolean and integer columns with values, 3. Interpolate NA values for real valued streams, 4. Drop first/last rows that contain NA for any of the streams. In the end, a new file, called cleaned_data.pkl, is generated.|
|DatasetCreation.py|Third part of preprocessing. Conversion of the cleaned data frames of all partial data sets into the training data.|
|DatasetPostProcessing.py | Additional, subsequent changes to a dataset are done by this script.|
|RealTimeClassification.py|Contains the implementation of the real time data processing.|
