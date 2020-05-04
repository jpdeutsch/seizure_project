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
%datasetPath = fullfile("E:", "School", "EE5549", "Detection");
datasetPath = fullfile("..","seizure_data");
%figurePath = fullfile("E:","School","EE5549","Detection","Figures");
figurePath = fullfile("..","Figures");

startPatient = 1; endPatient = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Preprocess the data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[trainPaths, trainLabels, testPaths, testLabels] = preprocess_detection_noTestData(datasetPath,...
    patients,startPatient,endPatient);

filterSize = [2];
numFilters = [32];
maxPool = [3];
dropout = [0.2]; 
numChannels = [4, 8];
%stftScenario = [1 2];

C = {filterSize,numFilters,maxPool,dropout,numChannels};
D = C;
[D{:}] = ndgrid(C{:});
scenarios = cell2mat(cellfun(@(m)m(:),D,'uni',0));
%%
for i=1:length(scenarios)
i
tic  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Put data into datastores for network
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[dsTrain, dsTest] = datastores_detection_noTestData(trainPaths,testPaths,...
    0,scenarios(i,5),classes,trainLabels,testLabels);
%[dsTrain, dsTest, dsTestLabel] = datastores_detection_noTestData(trainPaths,testPaths,...
%    0,0,classes);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Training the Network
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
inputSize = size(preview(dsTrain));

prog = "Starting Training"
net = buildNetwork_detection(dsTrain, inputSize, scenarios(i,1), scenarios(i,2), ...
    scenarios(i,3), scenarios(i,4));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Run the network on the test data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

prog = "Classifying"
[pred, scores] = classify(net, dsTest);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Evaluate Network Performance
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

evaluateResults(testLabels,pred,i,figurePath);
toc
end