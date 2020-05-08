function [dsTest,dsVal] = datastores(valPaths,valLabels,...
    testPaths,testLabels,numChannels)

% Datastore for testing data
dsTest = imageDatastore(testPaths,'FileExtensions','.mat','ReadFcn', @(f)...
    loadData(f,numChannels),'Labels',categorical(testLabels));

% Datastore for validation data
dsVal = imageDatastore(valPaths,'FileExtensions','.mat','ReadFcn', @(f)...
    loadData(f,numChannels),'Labels',categorical(valLabels));

end


function data = loadData(matFile,numChannels)

tmp = load(matFile); % struct loaded in from memory

% Downsample the data to 200Hz
freq = ceil(tmp.freq);
if freq ~= 200
    data = resample(tmp.data',200,freq)';
else 
    data = tmp.data;
end


% Calculate variance of each channel and only take the most variant
var =  (1/200)*sum((data-mean(data,2)).^2,2);
[~,idx] = sort(var,'descend');
data = data(idx(1:numChannels),:);

end