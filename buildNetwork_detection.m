
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Training the Network
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

net = train_network(dsTrain, dsVal, classes);

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Preprocessing the test data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

testData = prepTestData(datasetPath);

% Datastore for test data
fdsTest = fileDatastore(testData.fullPath',...
    'ReadFcn', @(fileName) loadTestData(fileName));

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Run the network on the test data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

pred = classify(net, fdsTest);


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Helper functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%{
% Given training and validation datastores, creates a neural network and
% trains the data, returning a trained network.
%
% Parameters:
%   dsTrain - Training data in datastore format. {sequence data} {label}
%   dsVal - Validation data in datastore format. {sequence data} {label}
%   classes - Categorical array of classes (labels)
% Output:
%   net - Trained neural network
%}
function net = train_network(dsTrain, dsVal, classes)

% TODO: use network builder to build network, two branches of image input
% [stft size, number based on window size, numChannels] -> conv -> relu ->
% maxpool -> dropout (order may be off) x3, only two dropouts (or maybe
% maxpools), then addition layer -> fully connected -> softmax -> class

options = trainingOptions('adam',...
    'ExecutionEnvironment', 'cpu',...
    'MaxEpochs', 500,...
    'ValidationData', dsVal,...
    'ValidationFrequency', 10,...
    'ValidationPatience', 5,...
    'GradientThreshold', 2,...
    'Shuffle', 'never',...
    'Verbose',0,...
    'Plots', 'training-progress');


net = trainNetwork(dsTrain, layers, options);
end