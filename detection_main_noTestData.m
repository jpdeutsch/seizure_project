clear
close all
clc

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Initial Processing of Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Classes of all data
classes = ["ictal", "interictal"];

% All patients in the study
patients = ["Dog_"+string(1:4), "Patient_"+string(1:8)];

% Base paths for seizure detection datasets of each patient
datasetPath = fullfile("D:", "School", "EE5549", "Detection");

startPatient = 1; endPatient = 2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Preprocess the data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[trainPaths, testPaths] = preprocess_detection_noTestData(datasetPath,...
    patients,startPatient,endPatient);

filterSize = [2, 4];
numFilters = [32, 64];
maxPool = [3, 6];
dropout = [0.2, 0.5]; 
numChannels = [4, 8];

C = {filterSize,numFilters,maxPool,dropout,numChannels};
D = C;
[D{:}] = ndgrid(C{:});
scenarios = cell2mat(cellfun(@(m)m(:),D,'uni',0));


for i=1:length(scenarios)
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Put data into datastores for network
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[dsTrain, dsTest, dsTestLabel] = datastores_detection_noTestData(trainPaths,testPaths,...
    0,scenarios(i,5),classes);

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Training the Network
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tmp = preview(dsTrain);
inputSize = size(tmp{1});

net = buildNetwork_detection(dsTrain, inputSize, scenarios(i,1), scenarios(i,2), ...
    scenarios(i,3), scenarios(i,4));

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Run the network on the test data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[pred, scores] = classify(net, dsTest);


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Evaluate Network Performance
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

evaluateResults(dsTestLabel,pred,i);

end