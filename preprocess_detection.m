function [trainPaths, valPaths, testData] = preprocess_detection(path,...
    patients,startPatient,endPatient)
% Fraction of data to be in training group
trainFrac = 3/4;

% Full path to each patient's data
pathPatients = fullfile(path, patients);

%TODO rename files so already in correct order and this isn't required
% File path for ictal data of all patients
ictal_data = sortFiles(pathPatients,...
    fullfile(pathPatients, "*_ictal*.mat"));
% File path for interictal data of all patients
interictal_data = sortFiles(pathPatients,...
    fullfile(pathPatients, "*_interictal*.mat"));

% TODO: use these numbers to create minibatches for training with a mixture
% of ictal and interictal data
% Array of the amount of each patient's ictal data segments
totalIctal = cellfun(@(c) length(c), ictal_data,'UniformOutput',true);
% Array of the amount of each patient's interictal data segments
totalInterictal = cellfun(@(c) length(c), interictal_data,...
    'UniformOutput',true);

% TODO: perform above getting the ictal data for each patient on the test
% data so can combine the test and training data, then split out test

% Arrays of the amount of each patient for training data
trainIctal = ceil(totalIctal.*trainFrac);
trainInterictal = ceil(totalInterictal.*trainFrac);

% TODO: hardcoded for just dogs right now, use startPatient and endPatient
% to preallocate array and speed things up
% Total ictal and interictal data for all patients
%trainingPath = strings(1,sum(trainIctal(1:4))+sum(trainInterictal(1:4)));
%validationPath = strings(1,sum(totalIctal(1:4)-trainIctal(1:4))+...
%    sum(totalInterictal(1:4)-trainInterictal(1:4)));
trainPaths = []; valPaths = [];

% TODO: hardcoded for just dogs right now
for i=startPatient:endPatient
    trainPaths = [trainPaths,...
        [ictal_data{1,i}{1:trainIctal(i)}],...
        [interictal_data{1,i}{1:trainInterictal(i)}]];
    valPaths = [valPaths,...
        [ictal_data{1,i}{trainIctal(i)+1:totalIctal(i)}],...
        [interictal_data{1,i}{trainInterictal(i)+1:totalInterictal(i)}]];
end

testData = prepTestData(path);

end

%{
% Manipulates test data labels to be used in testing the neural network.
%
% Parameters:
%   dataPath - Path to where the test .csv data is located
% Output:
%   testData - Table of modified test data to be used in network testing
%}
function testData = prepTestData(dataPath)
% Load data from .csv into Matlab
testData = readtable(fullfile(dataPath, "SzDetectionAnswerKey.csv"));

% TODO: hardcoded for just dog data right now
testData = testData(1:13641,:); % just to get only dog tests

% Extract patient info from file name
splitName = split(testData.clip, "_");
testData.patient = string(splitName(:,1))+"_"+string(splitName(:,2));

% Goes through each row of table and constructs full file path for data
testData.fullPath = rowfun(@(x,y) fullfile(dataPath,x,y), ...
    testData, 'InputVariables', {'patient', 'clip'}, ...
    'OutputFormat', 'uniform');

% Adds label name to table based on seizure column
testData.label(testData.seizure==1) = categorical("ictal");
testData.label(testData.seizure==-1) = categorical("interictal");
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