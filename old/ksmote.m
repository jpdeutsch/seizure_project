%% creating the data for python

data_reshaped = zeros(505,800);
for i=1:505
data_reshaped(i,1:200) = trainPaths(1,:,1,i);
data_reshaped(i,201:400) = trainPaths(2,:,1,i);
data_reshaped(i,401:600) = trainPaths(3,:,1,i);
data_reshaped(i,601:800) = trainPaths(4,:,1,i);
end

patientData.data = data_reshaped;
patientData.labels = double(trainLabels=="ictal");


% loading the data from python and transforming for use in the network

% load in the data, below x is the struct which is loaded in

trainPaths_ksmote = zeros(4,200,1,588);
>> for i = 1:588
trainPaths_ksmote(1,:,1,i) = x.data(i,1:200);
trainPaths_ksmote(2,:,1,i) = x.data(i,201:400);
trainPaths_ksmote(3,:,1,i) = x.data(i,401:600);
trainPaths_ksmote(4,:,1,i) = x.data(i,601:800);
end

trainLabels_ksmote = strings(588,1);
trainLabels_ksmote(x.labels==1) = "ictal";
trainLabels_ksmote(x.labels==0) = "interictal";
trainLabels_ksmote = categorical(trainLabels_ksmote);