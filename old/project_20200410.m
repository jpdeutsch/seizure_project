clear
close all
clc

% Classes of all data
classes = ["ictal", "interictal"];

% All patients in the study
patients = ["Dog_"+string(1:4), "Patient_"+string(1:8)];

% Base paths for seizure detection datasets of each patient
datasetPath = fullfile("E:", "School", "EE5549", "Detection");
pathPatients = fullfile(datasetPath, patients);


testAnswers = readtable(fullfile(datasetPath, "SzDetectionAnswerKey.csv"));
testAnswers.patient = addPatientToTable(testAnswers);
testAnswers.fullPath = rowfun(@(x,y) fullfile(datasetPath,x,y), ...
    testAnswers, 'InputVariables', {'patient', 'clip'}, ...
    'OutputFormat', 'uniform');


% File path for ictal data of all patients
ictal_data = sortFiles(pathPatients,...
    fullfile(pathPatients, "*_ictal*.mat"));
% File path for interictal data of all patients
interictal_data = sortFiles(pathPatients,...
    fullfile(pathPatients, "*_interictal*.mat"));

dog_ictal = [ictal_data{1,1:4}]; % all ictal dog file paths sorted
dog_interictal = [interictal_data{1,1:4}]; % interictal dog file paths sorted

% Divide data into training and validation groups
trainFrac = 3/4;
trainingPath = []; validationPath = [];
for i=1:4
    idx_ictal = ceil(trainFrac * length(ictal_data{1,i}));
    idx_interictal = ceil(trainFrac * length(interictal_data{1,i}));
    trainingPath = [trainingPath,...
        [ictal_data{1,i}{1:idx_ictal}],...
        [interictal_data{1,i}{1:idx_interictal}]];
    validationPath = [validationPath,...
        [ictal_data{1,i}{idx_ictal+1:length(ictal_data{1,i})}],...
        [interictal_data{1,i}{idx_interictal+1:length(interictal_data{1,i})}]];
end

% Datastore for training data
fdsTrain = fileDatastore(trainingPath,...
    'ReadFcn', @(fileName) loadData(fileName,classes));

% Datastore for validation data
fdsVal = fileDatastore(validationPath,...
    'ReadFcn', @(fileName) loadData(fileName,classes));


% Datastore for test data
fdsTest = fileDatastore(testAnswers.fullPath',...
    'ReadFcn', @(fileName) loadTestData(fileName,classes,testAnswers));

%fdsFreq = fileDatastore(trainingPath,...
%    'ReadFcn', @(fileName) loadFreq(fileName));
%{
fdsLabelTrain = fileDatastore(trainingPath,...
    'ReadFcn', @(fileName) readLabel(fileName, classes));
fdsLabelVal = fileDatastore(trainingPath,...
    'ReadFcn', @(fileName) readLabel(fileName, classes));
%}
%cdsTrain = combine(fdsTrain, fdsLabelTrain);
%cdsVal = combine(fdsVal, fdsLabelVal);

%net = train_network(fdsTrain, fdsVal, classes);

%% Helper functions
% Preprocess the training and validation data
function [dsTrain, dsVal] = preProcess()



end

% Load data from input struct of each .mat file
function data = loadData(matFile,classes)
data = cell(1,2);
tmp = load(matFile);

% Data from the .mat file
data(1,1) = {tmp.data};

[~,name] = fileparts(matFile);
name_parts = split(name, "_");
data(1,2) = {categorical(name_parts(3), classes)};
end

function data = loadTestData(matFile,classes,answers)
data = cell(1,2);
tmp = load(matFile);

data(1,1) = {tmp.data};
seizure = answers.seizure(find(strcmp(answers.fullPath,matFile),1));
if seizure == 1
    data(1,2) = {categorical(classes(1))};
else
    data(1,2) = {categorical(classes(2))};
end
end

function freq = loadFreq(matFile)
tmp = load(matFile);
freq = tmp.freq;
end

function label = readLabel(fileName, classNames)
[~,name] = fileparts(fileName);
name_parts = split(name, "_");
label = categorical(name_parts(3), classNames);
end

function label = getTestLabel(fileName)
%x = strcmp(testCorrect.clip, "Dog_1_test_segment_4.mat");
%p = rowfun(@(x) fullfile(datasetPath,x), testAnswers, 'InputVariables', {'clip'});
end

function patient = addPatientToTable(table)
splitName = split(table.clip,"_");
patient = string(splitName(:,1))+"_"+string(splitName(:,2));
end

function sorted_files = sortFiles(root, label)
S = arrayfun(@(n)dir(n), label, 'uni', 0);
N = cellfun(@(n)natsortfiles({n.name}), S, 'uni', 0);
sorted_files = cell(1,12);
for i = 1:12
sorted_files{i} = cellfun(@(n)fullfile(root(i),n),N{i},'uni',0);
end
end

function net = train_network(dsTrain, dsVal, classes)
%% scratch neural net
numFeatures = 16;
numClasses = numel(classes);
numHiddenUnits = 50;

layers = [...
    sequenceInputLayer(numFeatures)
    bilstmLayer(numHiddenUnits, 'OutputMode', 'last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

options = trainingOptions('adam',...
    'ExecutionEnvironment', 'cpu',...
    'MaxEpochs', 25,...
    'ValidationData', dsVal,...
    'ValidationFrequency', 10,...
    'GradientThreshold', 2,...
    'Shuffle', 'never',...
    'Verbose',0,...
    'Plots', 'training-progress');


net = trainNetwork(dsTrain, layers, options);
end