function net = buildNetwork_detection(dsTrain,dsVal,inputSize,...
    filterSize,numFilters,maxPool)  

layers = [
    imageInputLayer(inputSize,"Name","imageinput_1",'Normalization','none')
    convolution2dLayer([2 3],32,"Padding","same","Name","conv_1")
    batchNormalizationLayer("Name","batch_1")
    maxPooling2dLayer([maxPool maxPool],"Name","pool_1","Padding","same")
    reluLayer("Name","relu_1")
    convolution2dLayer([2 3],32,"Padding","same","Name","conv_2")
    batchNormalizationLayer("Name","batch_2")
    maxPooling2dLayer([maxPool maxPool],"Name","pool_2","Padding","same")
    reluLayer("Name","relu_2")
    convolution2dLayer([2 3],32,"Padding","same","Name","conv_3")
    batchNormalizationLayer("Name","batch_3")
    maxPooling2dLayer([maxPool maxPool],"Name","pool_3","Padding","same")
    reluLayer("Name","relu_3")
    convolution2dLayer([2 3],32,"Padding","same","Name","conv_4")
    batchNormalizationLayer("Name","batch_4")
    maxPooling2dLayer([maxPool maxPool],"Name","pool_4","Padding","same")
    fullyConnectedLayer(2,"Name","fc")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];   

options = trainingOptions('sgdm',...
    'ExecutionEnvironment', 'parallel',...
    'MaxEpochs', 35,...
    'MiniBatchSize',64,...
    'Shuffle', 'every-epoch',...
    'Verbose',1,...
    'VerboseFrequency',100,...
    'ValidationData',dsVal,...
    'ValidationFrequency',25,...
    'Momentum',0.5);


net = trainNetwork(dsTrain,layers,options);

end