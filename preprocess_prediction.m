function [trainPaths, trainLabels, testPaths, testLabels, valPaths, valLabels] = ...
    preprocess_detection(path,patient)
% Full path to patient data
patientPath = fullfile(path,patient,patient);

% Paths to ictal data and interictal data for this patient
[preictalPaths,interictalPaths] = prepPatient(patientPath);

% Paths to the ictal and interictal test data for this patient
[testPreictal,testInterictal] = prepTestData(path,patientPath,patient);
    
% Concatenate the ictal and interictal paths into one for all training data
trainPaths = vertcat(preictalPaths,interictalPaths);

% Generate labels for training data
trainLabels = vertcat(repmat("preictal",[length(preictalPaths) 1]),...
    repmat("interictal",[length(interictalPaths) 1]));

% Random indices in test ictal data to include in validation
valSplitPreictal = randperm(length(testPreictal));

% Random indices in test interictal data to include in validation
valSplitInterictal = randperm(length(testInterictal));

% Fraction of test data to include in validation
fracPreictalVal = floor(0.2*length(testPreictal));
fracInterictalVal = floor(0.2*length(testInterictal));

% Ictal validation data
valPreictal = testPreictal(valSplitPreictal(1:fracPreictalVal));

% Interictal validation data
valInterictal = testInterictal(valSplitInterictal(1:fracInterictalVal));

% Concatenate ictal and interictal into one validation path
valPaths = vertcat(valPreictal,valInterictal);

% Generate labels for validation data
valLabels = vertcat(repmat("preictal",[length(valPreictal),1]),...
    repmat("interictal",[length(valInterictal),1]));

% Take remainingg test data and set to testIctal/testInterictal
testPreictal = testPreictal(valSplitPreictal(fracPreictalVal+1:end));
testInterictal = testInterictal(valSplitInterictal(fracInterictalVal+1:end));

% Concatenate ictal and interictal test data and generate labels
testPaths = vertcat(testPreictal,testInterictal);
testLabels = vertcat(repmat("preictal",[length(testPreictal),1]),...
    repmat("interictal",[length(testInterictal) 1]));

end

function [preictalPaths,interictalPaths] = prepPatient(path)

% Get all ictal clips in patient directory
preictalClips = dir(fullfile(path,"*_preictal_*.mat"));

% Get all interictal clips in patient directory
interictalClips = dir(fullfile(path,"*_interictal_*.mat"));

% Get the full path for each ictal file
preictalPaths = arrayfun(@(f) fullfile(path,preictalClips(f).name),...
    [1:length(preictalClips)],'uni',false)';

% Get the full path for each interictal file
interictalPaths = arrayfun(@(f) fullfile(path,interictalClips(f).name),...
    [1:length(interictalClips)],'uni',false)';

%oversample ictal signals
%over_idx = randperm(length(preictalPaths),floor(0.2*length(preictalPaths)));
%preictalPaths = vertcat(preictalPaths,preictalPaths(over_idx));

%undersample interictal signals
%under_idx = randperm(length(interictalPaths),floor(0.7*length(interictalPaths)));
%interictalPaths = interictalPaths(under_idx);

end

%{
% Manipulates test data labels to be used in testing the neural network.
%
% Parameters:
%   dataPath - Path to where the test .csv data is located
% Output:
%   testData - Table of modified test data to be used in network testing
%}
function [testPreictal, testInterictal] = prepTestData(dataPath,patientPath,patient)
% Load data from .csv into Matlab
testData = readtable(fullfile(dataPath, "SzPrediction_answer_key.csv"),...
    'Delimiter',',');

% Get the test data just for this patient
patientTests = testData(contains(testData.clip,patient),:);

% Get the file name for this clip
[~, name, ~] = arrayfun(@(f) fileparts(patientTests.clip{f}),[1:size(patientTests)],...
    'uni',false);
splitName = split(name',"_");

%{
% Create column with the updated name using leading zeros in the number
patientTests.newName = arrayfun(@(f) [strjoin([splitName(f,1:end-1),...
    num2str(str2double(splitName{f,end}),'%04.f')],"_"), '.mat'],...
    (1:size(patientTests)),'uni',false)';
%}

% Paths to all ictal segments in the test data for this patient
testPreictal = fullfile(patientPath,...
    patientTests.clip(patientTests.preictal==1));
% Paths to all interictal segments in the test data for this patient
testInterictal = fullfile(patientPath,...
    patientTests.clip(patientTests.preictal==0));

end
