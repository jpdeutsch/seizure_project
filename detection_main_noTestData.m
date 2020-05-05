clear
close all
clc


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Initial Processing of Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Classes of all data
classes = ["ictal", "interictal"];

% All patients in the study
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

ri.patients = 3:6;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Preprocess the data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[trainPaths, trainLabels, testPaths, testLabels] = ...
    preprocess_detection_noTestData(datasetPath,patients,ri.patients);

filterSize = [2 4];
numFilters = [32 64];
maxPool = [2];
dropout = [0.5]; 
numChannels = [4];
stftScenario = [2];

% creates cell matrix for every trial version of network
C = {filterSize,numFilters,maxPool,dropout,numChannels,stftScenario};
D = C;
[D{:}] = ndgrid(C{:});
scenarios = cell2mat(cellfun(@(m)m(:),D,'uni',0));
ri.scenarios = scenarios;
save('../Figures/runInfo.mat','-struct','ri');
save('../all_data/runInfo.mat','-struct','ri');


%%
for i=1:size(scenarios,1)
i
tic  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Put data into datastores for network
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[dsTrain, dsTest] = datastores_detection_noTestData(trainPaths,testPaths,...
    0,scenarios(i,5),classes,trainLabels,testLabels);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Training the Network
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

inputSize = size(preview(dsTrain));
prog = "Starting Training"
net = buildNetwork_detection(dsTrain,inputSize,scenarios(i,1),scenarios(i,2),...
    scenarios(i,3),scenarios(i,4));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Run the network on the test data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

prog = "Classifying"
[pred, scores] = classify(net, dsTest);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Evaluate Network Performance
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

confusion_value = evaluateResults(testLabels,pred,i,figurePath)
toc
end