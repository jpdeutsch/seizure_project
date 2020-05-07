%function net = buildNetwork_detection(trainData,trainLabels,inputSize,...
%    filterSize,numFilters,maxPool,dropout)
function net = buildNetwork_detection(trainData,trainLabels,inputSize,...
    filterSize,numFilters,maxPool,dropout,dsVal)
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

   
    %{
    layers = [...
    sequenceInputLayer(inputSize)
    bilstmLayer(50, 'OutputMode', 'last')
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];

  
options = trainingOptions('sgdm',...
    'ExecutionEnvironment','cpu',...
    'MaxEpochs',300,...
    'MiniBatchSize',64,...
    'Shuffle','every-epoch',...
    'Verbose',0,...
    'VerboseFrequency',100,...
    'Plots','training-progress');
    %}

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


net = trainNetwork(trainData,layers,options);

end