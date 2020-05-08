function [dsTrain, dsVal, dsTest] = datastores_detection(trainPaths,valPaths,...
    testPaths,stftScenario,numChannels,classes)

% Datastore for training data
dsTrainReal = fileDatastore(trainPaths,'ReadFcn', @(fileName)...
    loadData(fileName,classes,"stft_real",stftScenario,numChannels));

dsTrainImag = fileDatastore(trainPaths,'ReadFcn', @(fileName)...
    loadData(fileName,classes,"stft_imag",stftScenario,numChannels));

dsTrainLabel = fileDatastore(trainPaths,'ReadFcn', @(fileName)...
    loadData(fileName,classes,"label",stftScenario,numChannels));

dsTrain = combine(dsTrainReal, dsTrainImag, dsTrainLabel);

% Datastore for validation data
dsValData = fileDatastore(valPaths,'ReadFcn', @(fileName)...
    loadData(fileName,classes,"stft",stftScenario,numChannels));

dsValLabel = fileDatastore(valPaths,'ReadFcn', @(fileName)...
    loadData(fileName,classes,"label",stftScenario,numChannels));

dsVal = combine(dsValData, dsValLabel);

% Datastore for test data
dsTestReal = fileDatastore(testPaths,'ReadFcn', @(fileName)...
    loadData(fileName,classes,"stft_real",stftScenario,numChannels));

dsTestImag = fileDatastore(testPaths,'ReadFcn', @(fileName)...
    loadData(fileName,classes,"stft_imag",stftScenario,numChannels));

dsTest = combine(dsTestReal,dsTestImag);

end

%{
% Load data from input struct of each .mat file for use in training and 
% validation datastores.
%
% Parameters:
%   matFile - Path to the current .mat file holding the data
%   classes - Array of the class names
% Outputs:
%   data - Cell vector of {sequence data} {category}
%}
function data = loadData(matFile,classes,field,stftScenario,numChannels)
% TODO: have switch statement for stftScenario, which will do different stft
%window lengths/types depending on scenario number, once decided will want
%to move to preprocess secion

tmp = load(matFile); % struct loaded in from memory

%TODO resample data down to 400 Hz if not already there, do ceil function
%on tmp.freq and check if 400, if not sample down using resample(
freq = ceil(tmp.freq);
if freq ~= 400
    data = resample(tmp.data',400,freq)';
else 
    data = tmp.data;
end

% TODO: use formula (1/400) sum((channel data - mean of channel)^2) to find
%variance of each channel, then pick numChannels of the highest variance
%channels and only use that data. May want to do this in preprocess section
%once number decided

if strcmp(field, "label")
    [~,name] = fileparts(matFile); % extract name of file without file ext
    name_parts = split(name, "_"); % split up name at underscores
    % Assigns category for data to relevant part of file name
    data = categorical(name_parts(3), classes);
elseif strcmp(field, "stft_real")
    data = real(stft(data',400));
elseif strcmp(field, "stft_imag")
    data = imag(stft(data',400));
elseif strcmp(field, "stft")
    tmpStft = stft(data',400);
    data = {real(tmpStft), imag(tmpStft)};
end

end