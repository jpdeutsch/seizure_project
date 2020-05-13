%{
% Builds the neural network and trains it.
% 
% input parameters:
%   trainData - N x 1 cell array of 8 x 200 matrices of segment data for training
%   trainLabels - N x 1 categorical array of corresponding labels for trainData
%   dsVal - datastore with validation files and corresponding labels
%   inputSize - the size of the matrix to the input layer
%   filterSize - size of the convolutional filter
%   numFilters - number of convolutional filters on each layer
%   maxPool - size of the max pooling
%
% outputs
%   net - the trained neural network
%}
function net = buildNetwork(trainData,trainLabels,dsVal,inputSize,...
    filterSize,numFilters,maxPool)  

% set the batch size of the network based on how many elements are in the training set
batchSize = floor(length(trainLabels)/15);

% the layers of the network
layers = [
    imageInputLayer(inputSize,"Name","imageinput_1",'Normalization','none')
    convolution2dLayer([1 8],64,"Padding","same","Name","conv_1")
    batchNormalizationLayer("Name","batch_1")
    maxPooling2dLayer([1 maxPool],"Name","pool_1","Padding","same")
    reluLayer("Name","relu_1")
    convolution2dLayer([1 filterSize],numFilters,"Padding","same","Name","conv_2")
    batchNormalizationLayer("Name","batch_2")
    maxPooling2dLayer([1 maxPool],"Name","pool_2","Padding","same")
    reluLayer("Name","relu_2")
    convolution2dLayer([1 filterSize],numFilters,"Padding","same","Name","conv_3")
    batchNormalizationLayer("Name","batch_3")
    maxPooling2dLayer([1 maxPool],"Name","pool_3","Padding","same")
    reluLayer("Name","relu_3")
    convolution2dLayer([1 filterSize],numFilters,"Padding","same","Name","conv_4")
    batchNormalizationLayer("Name","batch_4")
    maxPooling2dLayer([1 maxPool],"Name","pool_4","Padding","same")
    fullyConnectedLayer(2,"Name","fc")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];   

% The options for training the network
options = trainingOptions('adam',...
    'ExecutionEnvironment', 'parallel',...
    'MaxEpochs', 100,...
    'MiniBatchSize',batchSize,...
    'Shuffle', 'every-epoch',...
    'Verbose',1,...
    'VerboseFrequency',100,...
    'ValidationData',dsVal,...
    'ValidationFrequency',25,...
    'InitialLearnRate',0.001,...
    'L2Regularization',0.0005);


% train the network
net = trainNetwork(trainData,trainLabels,layers,options);

end
