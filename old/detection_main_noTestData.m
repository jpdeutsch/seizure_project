sclear
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
% Path for mounted drive
datasetPath = fullfile("..","all_data","Detection");

% Path for lab computer
figurePath = fullfile("..","Figures");

ri.patients = [3];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Preprocess the data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for p=ri.patients
    [trainPath,trainLabel,testPath,testLabel,valPaths,valLabels] = ...
        preprocess_detection(datasetPath,patients,ri.patients);
end

filterSize = [3];
numFilters = [32];
maxPool = [4];
dropout = [0.5]; 
numChannels = [4];
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
for i=1:size(scenarios,1)
i
tic  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Put data into datastores for network
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%[dsTrain, dsTest] = datastores_detection_noTestData(trainPaths,testPaths,...
%    trainLabels,testLabels);
[dsTrain, dsTest] = datastores_detection_noTestData(trainPaths,testPaths,...
    scenarios(i,6),scenarios(i,5),classes,trainLabels,testLabels);


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

confusion_value = evaluateResults(testLabels,pred,i,figurePath,scores)
toc
end