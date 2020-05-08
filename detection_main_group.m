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
% Path for mounted drive
datasetPath = fullfile("..","all_data","Detection");

% Path for lab computer
figurePath = fullfile("..","Figures");

ri.patients = [3];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Preprocess the data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
trainPaths = {}; trainLabels = {}; testPaths = {}; testLabels = {};
valPaths = {}; valLabels = {};
for p=ri.patients
    [trP,trL,tP,tL,vP,vL] = preprocess_detection(datasetPath,patients(p));
    trainPaths = vertcat(trainPaths,trP);
    trainLabels = vertcat(trainLabels,trL);
    testPaths = vertcat(testPaths,tP);
    testLabels = vertcat(testLabels,tL);
    valPaths = vertcat(valPaths,vP);
    valLabels = vertcat(valLabels,vL);
end

filterSize = [3];
numFilters = [32];
numChannels = [4];
stftScenario = [4];

% creates cell matrix for every trial version of network
C = {filterSize,numFilters,numChannels,stftScenario};
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
[dsTrain,dsTest,dsVal] = datastores_detection_noTestData(trainPaths,trainLabels,...
    valPaths,valLabels,testPaths,testLabels,scenarios(4));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Training the Network
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

inputSize = size(preview(dsTrain));
prog = "Starting Training"
net = buildNetwork_detection(dsTrain,dsVal,inputSize,scenarios(1),...
    scenarios(2),scenarios(3));


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