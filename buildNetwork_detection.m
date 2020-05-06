function net = buildNetwork_detection(dsTrain,inputSize,...
    filterSize,numFilters,maxPool,dropout)

%{
layers = [
    imageInputLayer(inputSize,"Name","imageinput_1",'Normalization','none')
    convolution2dLayer([filterSize filterSize],numFilters,"Name","conv_1","Padding","same")
    reluLayer("Name","relu_1")
    maxPooling2dLayer([maxPool maxPool],"Name","maxpool_1","Padding","same")
    dropoutLayer(dropout,"Name","dropout_1")
    convolution2dLayer([filterSize filterSize],numFilters,"Name","conv_2","Padding","same")
    reluLayer("Name","relu_2")
    maxPooling2dLayer([maxPool maxPool],"Name","maxpool_2","Padding","same")
    dropoutLayer(dropout,"Name","dropout_3")
    convolution2dLayer([filterSize filterSize],numFilters,"Name","conv_5","Padding","same")
    reluLayer("Name","relu_5")
    fullyConnectedLayer(2,"Name","fc")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];
%}
    
    
 
layers = [
    imageInputLayer(inputSize,"Name","imageinput_1",'Normalization','none')
    convolution2dLayer([filterSize filterSize],32,"Name","conv_1","Padding","same")
    reluLayer("Name","relu_1")
    convolution2dLayer([filterSize filterSize],64,"Name","conv_2","Padding","same")
    reluLayer("Name","relu_2")
    maxPooling2dLayer([maxPool maxPool],"Name","maxpool_2","Padding","same")
    dropoutLayer(dropout,"Name","dropout_3")
    fullyConnectedLayer(2,"Name","fc")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];   
    

options = trainingOptions('sgdm',...
    'ExecutionEnvironment', 'parallel',...
    'MaxEpochs', 30,...
    'MiniBatchSize',128,...
    'Shuffle', 'every-epoch',...
    'Verbose',1,...
    'VerboseFrequency',100,...
    'L2Regularization',0.0005,...
    'Momentum',0.9,...
    'Plots','training-progress');

net = trainNetwork(dsTrain, layers, options);

end