function [trainData, trainLabels, testPaths, testLabels, valPaths, valLabels] = ...
    preprocess(path,KSMOTEpath,patient,numChannels,downsample,useKSMOTE)

% Full path to patient data
patientPath = fullfile(path,patient);

if ~useKSMOTE
    % ictal data and interictal data for this patient
    [trainData,trainLabels] = prepPatientTrain(patientPath,downsample,numChannels);
else
    [trainData,trainLabels] = ksmoteLoad(fullfile(KSMOTEpath,...
        strcat(patient,"_ksmote.mat")),numChannels,downsample);
end


% Paths to the ictal and interictal test data for this patient
[testIctal,testInterictal] = prepTestData(path,patientPath,patient);

[valPaths,valLabels,testPaths,testLabels] = splitVal(testIctal,...
    testInterictal,0.2);


end


%%
%{
% Manipulates test data labels to be used in testing the neural network.
%
% Parameters:
%   dataPath - Path to where the test .csv data is located
% Output:
%   testData - Table of modified test data to be used in network testing
%}
function [testIctal, testInterictal] = prepTestData(dataPath,patientPath,patient)
% Load data from .csv into Matlab
testData = readtable(fullfile(dataPath, "SzDetectionAnswerKey.csv"));

% Get the test data just for this patient
patientTests = testData(contains(testData.clip,patient),:);

% Get the file name for this clip
[~, name, ~] = arrayfun(@(f) fileparts(patientTests.clip{f}),[1:size(patientTests)],...
    'uni',false);
splitName = split(name',"_");

% Create column with the updated name using leading zeros in the number
patientTests.newName = arrayfun(@(f) [strjoin([splitName(f,1:end-1),...
    num2str(str2double(splitName{f,end}),'%04.f')],"_"), '.mat'],...
    (1:size(patientTests)),'uni',false)';

% Paths to all ictal segments in the test data for this patient
testIctal = fullfile(patientPath,...
    patientTests.newName(patientTests.seizure==1));
% Paths to all interictal segments in the test data for this patient
testInterictal = fullfile(patientPath,...
    patientTests.newName(patientTests.seizure==-1));

end


%%
%TODO make sure validation data doesn't come after(?) test data
function [valPaths,valLabels,testPaths,testLabels] = splitVal(testIctal,...
    testInterictal,frac)

% Random indices in test ictal data to include in validation
%valSplitIctal = randperm(length(testIctal));
valSplitIctal = floor(frac*length(testIctal));

% Random indices in test interictal data to include in validation
%valSplitInterictal = randperm(length(testInterictal));
valSplitInterictal = floor(frac*length(testInterictal));

% Fraction of test data to include in validation
%fracIctalVal = floor(frac*length(testIctal));
%fracInterictalVal = floor(frac*length(testInterictal));

% Ictal validation data
%valIctal = testIctal(valSplitIctal(1:fracIctalVal));
valIctal = testIctal(1:valSplitIctal);

% Interictal validation data
%valInterictal = testInterictal(valSplitInterictal(1:fracInterictalVal));
valInterictal = testInterictal(1:valSplitInterictal);

% Concatenate ictal and interictal into one validation path
valPaths = vertcat(valIctal,valInterictal);

% Generate labels for validation data
valLabels = vertcat(repmat("ictal",[length(valIctal),1]),...
    repmat("interictal",[length(valInterictal),1]));

% Take remainingg test data and set to testIctal/testInterictal
%testIctal = testIctal(valSplitIctal(fracIctalVal+1:end));
testIctal = testIctal(valSplitIctal+1:end);
%testInterictal = testInterictal(valSplitInterictal(fracInterictalVal+1:end));
testInterictal = testInterictal(valSplitInterictal+1:end);

% Concatenate ictal and interictal test data and generate labels
testPaths = vertcat(testIctal,testInterictal);
testLabels = vertcat(repmat("ictal",[length(testIctal),1]),...
    repmat("interictal",[length(testInterictal) 1]));

end

%%
function [trainData, trainLabels] = ksmoteLoad(path,numChannels,downsample)
% loading the data from python and transforming for use in the network

tmp = load(path);
numSamples = size(tmp.data,1);

trainData = zeros(numChannels,downsample,1,numSamples);
for i = 1:size(tmp.data,1)
    for j =1:numChannels
        start = ((j-1)*downsample)+1;
        stop = j*downsample;
        trainData(j,:,1,i) = tmp.data(i,start:stop);
    end
end

trainLabels = strings(numSamples,1);
trainLabels(tmp.labels==1) = "ictal";
trainLabels(tmp.labels==0) = "interictal";
trainLabels = categorical(trainLabels);

end

