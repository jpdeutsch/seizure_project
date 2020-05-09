%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Initial Processing of Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Classes of all data
classes = ["ictal", "interictal"];

% All patients in the stu
patients = ["Dog_"+string(1:4), "Patient_"+string(1:8)];

% Base paths for seizure detection datasets of each patient
% Path for mounted drive
%datasetPath = fullfile("..","all_data","Detection");
datasetPath = "E:\School\EE5549\Detection";
%ksmotePath = fullfile(datasetPath,"ksmote");
ksmotePath = "E:\School\EE5549\Detection\ksmote";

% Path for lab computer
figurePath = fullfile("..","Figures");

patientToRun = 5;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Preprocess the data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

filterSize = [3];
numFilters = [32];
maxPool = [2];
dropout = [0.5]; 
numChannels = [8];

downsample = 200;

% creates cell matrix for every trial version of network
C = {filterSize,numFilters,maxPool,dropout,numChannels};
D = C;
[D{:}] = ndgrid(C{:});
scenarios = cell2mat(cellfun(@(m)m(:),D,'uni',0));
ri.scenarios = scenarios;

numRuns = 5;
conf = zeros(2,2,numRuns);
sens = zeros(1,5);
spec = zeros(1,5);
AUC = zeros(1,5);

%%
for i=1:numRuns
i


[trainData, trainLabels, testPaths, testLabels,valPaths,valLabels] = ...
    preprocess(datasetPath,ksmotePath,patients(p),numChannels,downsample,1);

tic  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Put data into datastores for network
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[dsTest,dsVal] = datastores(valPaths,valLabels,testPaths,testLabels,numChannels);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Training the Network
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

inputSize = [size(trainData,1), size(trainData,2)];
prog = "Starting Training"

net = buildNetwork(trainData,trainLabels,dsVal,inputSize,filterSize,numFilters,...
    maxPool);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Run the network on the test data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

prog = "Classifying"
[pred, scores] = classify(net, dsTest);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Evaluate Network Performance
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[conf(:,:,i),sens(i),spec(i),AUC(i)] = ...
    evaluateResults(testLabels,pred,figurePath,scores);
toc
end


average_TP = mean(conf(1,1,:))
average_FP = mean(conf(1,2,:))
average_FN = mean(conf(2,1,:))
average_TN = mean(conf(2,2,:))

average_sens = average_TP/(average_TP+average_FN)
average_spec = average_TN/(average_TN+average_FP)

average_AUC = mean(AUC)