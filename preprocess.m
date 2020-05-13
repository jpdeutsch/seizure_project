%{
%   Given the paths to the data, load it and format it correctly for the network
%   
%   inputs
%       path - path to the original data
%       KSMOTEpath - path to the KSMOTE data
%       patient - the patient to load
%       numChannels - number of channels to train on
%       downsample - frequency to downsample the data to
%       useKSMOTE - whether to use the KSMOTE data
%
%   outputs
%       trainData - data to train the network with
%       trainLabels - labels for the training data
%       testPaths - paths to the testing data
%       testLabels - labels for the testing data
%       valPaths - paths to the validation data
%       valLabels - labels for the validation data
%}
function [trainData, trainLabels, testPaths, testLabels, valPaths, valLabels] = ...
    preprocess(path,KSMOTEpath,patient,numChannels,downsample,useKSMOTE)

% Full path to patient data
patientPath = fullfile(path,patient);

% Either load kSMOTE data or raw data
if ~useKSMOTE
    % load the data for the given patient
    [trainData,trainLabels] = prepPatientTrain(patientPath,downsample,numChannels);
else
    % load the ksmote data
    [trainData,trainLabels] = ksmoteLoad(fullfile(KSMOTEpath,...
        strcat(patient,"_ksmote_",string(numChannels),"chan.mat")),numChannels,downsample);
end

% Paths to the ictal and interictal test data for this patient
[testIctal,testInterictal] = prepTestData(path,patientPath,patient);

% Paths to the validation and test data and corresponding labels
[valPaths,valLabels,testPaths,testLabels] = splitVal(testIctal,...
    testInterictal,0.2);

end


%%
%{
%   Preps the test data with labels from the answer key
%
%   inputs
%       dataPath - Path where test labels are stored
%       patientPath - Path to patient test segments
%       patient - which patient to prep test data for
%
%   ouputs
%       testIctal - ictal testing data
%       testInterictal - interictal testing data
%}
function [testIctal, testInterictal] = prepTestData(dataPath,patientPath,patient)

% Load data from .csv into Matlab
testData = readtable("SzDetectionAnswerKey.csv");

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
%{
%   function which splits up the testing data segments into validation and test
%
%   inputs
%       testIctal - ictal testing data
%       testInterictal - interictal testing data
%       frac - fraction of testing data to use for validation
%
%   outputs
%       valPaths - paths to validation data
%       valLabels - labels for validation data
%       testPaths - paths to testing data
%       testLabels - labels for testing data
%}
function [valPaths,valLabels,testPaths,testLabels] = splitVal(testIctal,...
    testInterictal,frac)

% grab the first fraction of ictal segments and use for validation
valSplitIctal = floor(frac*length(testIctal));

% grab the first fraction of interictal segments and use for validation
valSplitInterictal = floor(frac*length(testInterictal));

% Ictal validation data
valIctal = testIctal(1:valSplitIctal);

% Interictal validation data
valInterictal = testInterictal(1:valSplitInterictal);

% Concatenate ictal and interictal into one validation path
valPaths = vertcat(valIctal,valInterictal);

% Generate labels for validation data
valLabels = vertcat(repmat("ictal",[length(valIctal),1]),...
    repmat("interictal",[length(valInterictal),1]));

% Take remainingg test data and set to testIctal/testInterictal
testIctal = testIctal(valSplitIctal+1:end);
testInterictal = testInterictal(valSplitInterictal+1:end);

% Concatenate ictal and interictal test data and generate labels
testPaths = vertcat(testIctal,testInterictal);
testLabels = vertcat(repmat("ictal",[length(testIctal),1]),...
    repmat("interictal",[length(testInterictal) 1]));

end

%%
%{
%   loads ksmote data from python for use in the network 
%
%   inputs
%       path - path to the ksmote data
%       numChannels - number of channels in the data
%       downsample - frequency of the data
%
%   outputs
%       trainData - data to train the network on
%       trainLabels - labels for the training data
%}
function [trainData, trainLabels] = ksmoteLoad(path,numChannels,downsample)
% loading the data from python and transforming for use in the network

% Load in the struct
tmp = load(path);

% number of segments in the training data
numSamples = size(tmp.data,1);

% change data to the correct format
trainData = zeros(numChannels,downsample,1,numSamples);
for i = 1:size(tmp.data,1)
    for j =1:numChannels
        start = ((j-1)*downsample)+1;
        stop = j*downsample;
        trainData(j,:,1,i) = tmp.data(i,start:stop);
    end
end

% Generate labels for the training data
trainLabels = strings(numSamples,1);
trainLabels(tmp.labels==1) = "ictal";
trainLabels(tmp.labels==0) = "interictal";
trainLabels = categorical(trainLabels);

end

