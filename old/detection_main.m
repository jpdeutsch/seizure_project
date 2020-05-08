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

startPatient = 1; endPatient = 4;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Preprocess the data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[trainPaths, testPaths, testData] = preprocess_detection(datasetPath,...
    patients,startPatient,endPatient);
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Put data into datastores for network
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[dsTrain, dsVal, dsTest] = datastores_detection(trainPaths,testPaths,...
    testData.fullPath',0,8,classes);

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Training the Network
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% TODO: allow train network to take in different filter sizes, filter
% numbers, maxpool sizes, and dropout

% TODO: get inputSize from first data in dsTrain

net = train_network(dsTrain, dsVal, classes, inputSize);

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Run the network on the test data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

pred = classify(net, fdsTest);


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Evaluate Network Performance
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

accuracy = sum(pred == testData.label)./numel(pred);

% Plot ROC Curve
actual = [(testData.label=="ictal")'; (testData.label=="interictal")'];
predicted = [(pred=="ictal")'; (pred=="interictal")'];
plotroc(actual, predicted);
axesUserData=get(gca,'userdata');
legend(axesUserData.lines,'ictal', 'interictal');

% Calculate Confusion Matrix
[c, cm, ind, per] = confusion(int8(actual), int8(predicted));
plotconfusion(int8(actual),int8(predicted));
