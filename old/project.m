clear
close all
clc

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Initial Processing of Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Classes of all data
classes = ["ictal", "interictal"];

% All patients in the study
patients = ["Dog_"+string(1:4), "Patient_"+string(1:8)];

% Base paths for seizure detection datasets of each patient
datasetPath = fullfile("E:", "School", "EE5549", "Detection");

[dsTrain, dsVal] = preProcess(datasetPath, patients, classes);

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Training the Network
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

net = train_network(dsTrain, dsVal, classes);

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Preprocessing the test data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

testData = prepTestData(datasetPath);

% Datastore for test data
fdsTest = fileDatastore(testData.fullPath',...
    'ReadFcn', @(fileName) loadTestData(fileName));

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Run the network on the test data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

pred = classify(net, fdsTest);


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Evaluate Network Performance
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

accuracy = sum(pred == testData.label)./numel(pred)

% Plot ROC Curve
actual = [(testData.label=="ictal")'; (testData.label=="interictal")'];
predicted = [(pred=="ictal")'; (pred=="interictal")'];
plotroc(actual, predicted);
axesUserData=get(gca,'userdata');
legend(axesUserData.lines,'ictal', 'interictal');

% Calculate Confusion Matrix
[c, cm, ind, per] = confusion(int8(actual), int8(predicted));
plotconfusion(int8(actual),int8(predicted));


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Helper functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Preprocess the training and validation data
function [dsTrain, dsVal] = preProcess(path, patients, classes)
% Full path to each patient's data
pathPatients = fullfile(path, patients);

% File path for ictal data of all patients
ictal_data = sortFiles(pathPatients,...
    fullfile(pathPatients, "*_ictal*.mat"));
% File path for interictal data of all patients
interictal_data = sortFiles(pathPatients,...
    fullfile(pathPatients, "*_interictal*.mat"));

% Fraction of data to be in training group
trainFrac = 3/4;

% Array of the amount of each patient's ictal data segments
totalIctal = cellfun(@(c) length(c), ictal_data,'UniformOutput',true);
% Array of the amount of each patient's interictal data segments
totalInterictal = cellfun(@(c) length(c), interictal_data,...
    'UniformOutput',true);

% Arrays of the amount of each patient for training data
trainIctal = ceil(totalIctal.*trainFrac);
trainInterictal = ceil(totalInterictal.*trainFrac);

% TODO: hardcoded for just dogs right now
% Total ictal and interictal data for all patients
%trainingPath = strings(1,sum(trainIctal(1:4))+sum(trainInterictal(1:4)));
%validationPath = strings(1,sum(totalIctal(1:4)-trainIctal(1:4))+...
%    sum(totalInterictal(1:4)-trainInterictal(1:4)));
trainingPath = []; validationPath = [];

% TODO: hardcoded for just dogs right now
for i=1:1
    trainingPath = [trainingPath,...
        [ictal_data{1,i}{1:trainIctal(i)}],...
        [interictal_data{1,i}{1:trainInterictal(i)}]];
    validationPath = [validationPath,...
        [ictal_data{1,i}{trainIctal(i)+1:totalIctal(i)}],...
        [interictal_data{1,i}{trainInterictal(i)+1:totalInterictal(i)}]];
end

% Datastore for training data
dsTrainReal = fileDatastore(trainingPath,...
    'ReadFcn', @(fileName) loadData(fileName,classes,"stft_real"));

dsTrainImag = fileDatastore(trainingPath,...
    'ReadFcn', @(fileName) loadData(fileName,classes,"stft_imag"));

dsTrainLabel = fileDatastore(trainingPath,...
    'ReadFcn', @(fileName) loadData(fileName,classes,"label"));

dsTrain = combine(dsTrainReal, dsTrainImag, dsTrainLabel);

% Datastore for validation data
dsValData = fileDatastore(validationPath,...
    'ReadFcn', @(fileName) loadData(fileName,classes,"stft"));

dsValLabel = fileDatastore(validationPath,...
    'ReadFcn', @(fileName) loadData(fileName,classes,"label"));

dsVal = combine(dsValData, dsValLabel);

end

%{
% Load data from input struct of each .mat file for use in training and 
% validation datastores.
%
% Parameters:
%   matFile - Path to the current .mat file holding the data
%   classes - Array of the class names
% Outputs:
%   data - Cell vector of {sequence data} {category}
%}
function data = loadData(matFile,classes,field)
%data = cell(1,2); % empty data cell array
tmp = load(matFile); % struct loaded in from memory

if strcmp(field, "raw")
    data = tmp.data;
elseif strcmp(field, "label")
    [~,name] = fileparts(matFile); % extract name of file without file ext
    name_parts = split(name, "_"); % split up name at underscores
    % Assigns category for data to relevant part of file name
    data = categorical(name_parts(3), classes);
elseif strcmp(field, "freq")
    data = tmp.freq;
elseif strcmp(field, "stft_real")
    data = real(stft(tmp.data',ceil(tmp.freq)));
elseif strcmp(field, "stft_imag")
    data = imag(stft(tmp.data',ceil(tmp.freq)));
elseif strcmp(field, "stft")
    tmpStft = stft(tmp.data',ceil(tmp.freq));
    data = {real(tmpStft), imag(tmpStft)};
end
%data(1,1) = {tmp.data};
%data(1,1) = {stft(tmp.data',ceil(tmp.freq))}; % sequence data from struct




end

%{
% Loads data from testing matlab files into correct format for datastore
%
% Parameters:
%   matFile - filepath to the testing data
% Output:
%   data - data field of the struct
%}
function data = loadTestData(matFile)
tmp = load(matFile);
data = {tmp.data};
end

%{
% Manipulates test data labels to be used in testing the neural network.
%
% Parameters:
%   dataPath - Path to where the test .csv data is located
% Output:
%   correct - Table of modified test data to be used in network testing
%}
function correct = prepTestData(dataPath)
% Load data from .csv into Matlab
correct = readtable(fullfile(dataPath, "SzDetectionAnswerKey.csv"));

% TODO: hardcoded for just dog data right now
correct = correct(1:13641,:); % just to get only dog tests

% Extract patient info from file name
splitName = split(correct.clip, "_");
correct.patient = string(splitName(:,1))+"_"+string(splitName(:,2));

% Goes through each row of table and constructs full file path for data
correct.fullPath = rowfun(@(x,y) fullfile(dataPath,x,y), ...
    correct, 'InputVariables', {'patient', 'clip'}, ...
    'OutputFormat', 'uniform');

% Adds label name to table based on seizure column
correct.label(correct.seizure==1) = categorical("ictal");
correct.label(correct.seizure==-1) = categorical("interictal");
end

%{
% Given the root folders for each patient, sorts the folder names into
% the correct sequential order.
% 

%}
function sorted_files = sortFiles(root, label)
% Gets files in each subdirectory of the root data folder
S = arrayfun(@(n)dir(n), label, 'uni', 0);

% Sorts file names of each subdirectory
N = cellfun(@(n)natsortfiles({n.name}), S, 'uni', 0);

% From the sorted file names, reconstruct the entire datapath for each
sorted_files = cell(1,12); % initialize empty cell array for full file path
for i = 1:12
    sorted_files{i} = cellfun(@(n)fullfile(root(i),n),N{i},'uni',0);
end
end

%{
% Given training and validation datastores, creates a neural network and
% trains the data, returning a trained network.
%
% Parameters:
%   dsTrain - Training data in datastore format. {sequence data} {label}
%   dsVal - Validation data in datastore format. {sequence data} {label}
%   classes - Categorical array of classes (labels)
% Output:
%   net - Trained neural network
%}
function net = train_network(dsTrain, dsVal, classes)
% scratch neural net
numFeatures = [128,9,16]; % number of input features (channels)
numClasses = numel(classes); % number of possible output classifications
numHiddenUnits = 50; % parameter of lstm networks


layers = [...
    sequenceInputLayer(numFeatures, 'Name', 'input')
    
    flattenLayer('Name', 'flatten')
    lstmLayer(50,'OutputMode', 'sequence', 'Name', 'lstm1')
    dropoutLayer(0.2)
    
    lstmLayer(20,'OutputMode', 'last', 'Name', 'lstm2')
    dropoutLayer(0.2)
    
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer
    ];


%{
layers = [...
    imageInputLayer([16, 400, 1], 'Name', 'input')
    
    convolution2dLayer([1, 20], 32, 'Name', 'conv1')
    reluLayer('Name', 'relu1')
    maxPooling2dLayer([1, 3], 'Name', 'pooling1')
    
    convolution2dLayer([1, 20], 32, 'Name', 'conv2')
    reluLayer('Name', 'relu2')
    
    fullyConnectedLayer(2, 'Name', 'fullyConn')

    softmaxLayer('Name', 'softmax')
    
    classificationLayer('Name', 'classification')
    ];


%lgraph = layerGraph(layers);
%lgraph = connectLayers(lgraph, 'fold/miniBatchSize', 'unfold/miniBatchSize');
%}

options = trainingOptions('adam',...
    'ExecutionEnvironment', 'cpu',...
    'MaxEpochs', 500,...
    'ValidationData', dsVal,...
    'ValidationFrequency', 10,...
    'ValidationPatience', 5,...
    'GradientThreshold', 2,...
    'Shuffle', 'never',...
    'Verbose',0,...
    'Plots', 'training-progress');


net = trainNetwork(dsTrain, layers, options);
end