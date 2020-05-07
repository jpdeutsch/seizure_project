clear
close all
clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Initial Processing of Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Classes of all data
classes = ["ictal", "interictal"];

% All patients in the stu
patients = ["Dog_"+string(1:4), "Patient_"+string(1:8)];

% Base paths for seizure detection datasets of each patient
% Path for mounted drive
datasetPath = fullfile("..","all_data");

% Path for lab computer
figurePath = fullfile("..","Figures");

ri.patients = [7];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Preprocess the data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

filterSize = [3];
numFilters = [32];
maxPool = [2];
dropout = [0.5]; 
numChannels = [12];
stftScenario = [4];

% creates cell matrix for every trial version of network
C = {filterSize,numFilters,maxPool,dropout,numChannels,stftScenario};
D = C;
[D{:}] = ndgrid(C{:});
scenarios = cell2mat(cellfun(@(m)m(:),D,'uni',0));
ri.scenarios = scenarios;
save('../Figures/runInfo.mat','-struct','ri');
save('../all_data/runInfo.mat','-struct','ri');


%%
for p=ri.patients
i=1;
p
[trainPaths, trainLabels, testPaths, testLabels,valPaths,valLabels] = ...
    preprocess_detection(datasetPath,patients(ri.patients(p)));

tic  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Put data into datastores for network
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[dsTrain, dsTest,dsVal] = datastores_detection_noTestData(trainPaths,trainLabels,...
    valPaths,valLabels,testPaths,testLabels,numChannels);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Training the Network
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

inputSize = size(preview(dsTrain));
prog = "Starting Training"

net = buildNetwork_detection(dsTrain,dsVal,inputSize,filterSize,numFilters,...
    maxPool);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Run the network on the test data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

prog = "Classifying"
[pred, scores] = classify(net, dsTest);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Evaluate Network Performance
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

confusion_value = evaluateResults(testLabels,pred,p,figurePath,scores)
toc
end