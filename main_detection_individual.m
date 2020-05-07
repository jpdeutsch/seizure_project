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
% Path on Local Machine
%datasetPath = fullfile("E:", "School", "EE5549", "Detection");
% Path for data transferred to lab computer
%datasetPath = fullfile("..","seizure_data");
% Path for mounted drive
datasetPath = fullfile("..","all_data");

% Path on Local Machine
%figurePath = fullfile("E:","School","EE5549","Detection","Figures");
% Path for lab computer
figurePath = fullfile("..","Figures");

ri.patients = [3];

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
for p=1:length(ri.patients)
i=1;

[trainPaths, trainLabels, testPaths, testLabels,valPaths,valLabels] = ...
    preprocess_detection(datasetPath,patients,ri.patients(p),numChannels);

tic  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Put data into datastores for network
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%[dsTrain, dsTest] = datastores_detection_noTestData(trainPaths,testPaths,...
%    trainLabels,testLabels);
[dsTrain, dsTest,dsVal] = datastores_detection_noTestData(trainPaths,testPaths,...
    scenarios(i,6),scenarios(i,5),classes,trainLabels,testLabels,valPaths,valLabels);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Training the Network
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

inputSize = size(preview(dsTrain));
prog = "Starting Training"

net = buildNetwork_detection(dsTrain,trainLabels,inputSize,filterSize,numFilters,...
    maxPool,dropout,dsVal);


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