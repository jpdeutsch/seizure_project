function net = buildNetwork_detection(dsTrain,inputSize,...
    filterSize,numFilters,maxPool,dropout)

lgraph = layerGraph();

tempLayers = [
    imageInputLayer(inputSize,"Name","imageinput_1")
    convolution2dLayer([filterSize filterSize],numFilters,"Name","conv_1","Padding","same")
    reluLayer("Name","relu_1")
    maxPooling2dLayer([maxPool maxPool],"Name","maxpool_1","Padding","same")
    dropoutLayer(dropout,"Name","dropout_1")
    convolution2dLayer([filterSize filterSize],numFilters,"Name","conv_2","Padding","same")
    reluLayer("Name","relu_2")
    maxPooling2dLayer([maxPool maxPool],"Name","maxpool_2","Padding","same")
    dropoutLayer(dropout,"Name","dropout_3")
    convolution2dLayer([filterSize filterSize],numFilters,"Name","conv_5","Padding","same")
    reluLayer("Name","relu_5")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    imageInputLayer(inputSize,"Name","imageinput_2")
    convolution2dLayer([filterSize filterSize],numFilters,"Name","conv_3","Padding","same")
    reluLayer("Name","relu_3")
    maxPooling2dLayer([maxPool maxPool],"Name","maxpool_3","Padding","same")
    dropoutLayer(dropout,"Name","dropout_2")
    convolution2dLayer([filterSize filterSize],numFilters,"Name","conv_4","Padding","same")
    reluLayer("Name","relu_4")
    maxPooling2dLayer([maxPool maxPool],"Name","maxpool_4","Padding","same")
    dropoutLayer(dropout,"Name","dropout_4")
    convolution2dLayer([filterSize filterSize],numFilters,"Name","conv_6","Padding","same")
    reluLayer("Name","relu_6")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition")
    fullyConnectedLayer(2,"Name","fc")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];
lgraph = addLayers(lgraph,tempLayers);

lgraph = connectLayers(lgraph,"relu_5","addition/in1");
lgraph = connectLayers(lgraph,"relu_6","addition/in2");

clear tempLayers;

options = trainingOptions('adam',...
    'ExecutionEnvironment', 'cpu',...
    'MaxEpochs', 350,...
    'GradientThreshold', 2,...
    'Shuffle', 'never',...
    'Verbose',0);


net = trainNetwork(dsTrain, lgraph, options);
end