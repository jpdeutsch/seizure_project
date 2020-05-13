%{
%   Load test and validation data into datastores
%
%   inputs
%       valPaths - paths to all validation data
%       valLabels - labels for all validation data
%       testPaths - paths to all testing data
%       testLabels - labels for all validation data
%       numChannels - numbr of channels in the data
%
%   outputs
%}
function [dsTest,dsVal] = datastores(valPaths,valLabels,...
    testPaths,testLabels,numChannels)

% Datastore for testing data
dsTest = imageDatastore(testPaths,'FileExtensions','.mat','ReadFcn', @(f)...
    loadData(f,numChannels),'Labels',categorical(testLabels));

% Datastore for validation data
dsVal = imageDatastore(valPaths,'FileExtensions','.mat','ReadFcn', @(f)...
    loadData(f,numChannels),'Labels',categorical(valLabels));

end

%{
%   Function used by the datastores to load the data in the correct format
%}
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
