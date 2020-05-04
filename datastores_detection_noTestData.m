function [dsTrain, dsTest, dsTestLabel] = datastores_detection_noTestData(trainPaths,...
    testPaths,stftScenario,numChannels,classes,trainLabels,testLabels)

dsTrain = imageDatastore(trainPaths,'FileExtensions','.mat','ReadFcn', @(f)...
    loadData(f,classes,"stft_real",stftScenario,numChannels),...
    'Labels',categorical(trainLabels));
%{
% Datastore for training data
dsTrainReal = fileDatastore(trainPaths,'ReadFcn', @(fileName)...
    loadData(fileName,classes,"stft_real",stftScenario,numChannels));

dsTrainImag = fileDatastore(trainPaths,'ReadFcn', @(fileName)...
    loadData(fileName,classes,"stft_imag",stftScenario,numChannels));

dsTrainLabel = fileDatastore(trainPaths,'ReadFcn', @(fileName)...
    loadData(fileName,classes,"label",stftScenario,numChannels));

dsTrain = combine(dsTrainReal, dsTrainImag, dsTrainLabel);
%dsTrain = combine(dsTrainReal, dsTrainLabel);
%}
    
%{
% Datastore for test data
dsTestReal = fileDatastore(testPaths,'ReadFcn', @(fileName)...
    loadData(fileName,classes,"stft_real",stftScenario,numChannels));

dsTestImag = fileDatastore(testPaths,'ReadFcn', @(fileName)...
    loadData(fileName,classes,"stft_imag",stftScenario,numChannels));

dsTestLabel = fileDatastore(testPaths,'ReadFcn', @(fileName)...
    loadData(fileName,classes,"label",stftScenario,numChannels));

dsTest = combine(dsTestReal,dsTestImag);
%dsTest = dsTestReal;
%}
dsTest = imageDatastore(testPaths,'FileExtensions','.mat','ReadFcn', @(f)...
    loadData(f,classes,"stft_real",stftScenario,numChannels),...
    'Labels',categorical(testLabels));

dsTestLabel = 0;
    
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

freq = ceil(tmp.freq);
if freq ~= 400
    data = resample(tmp.data',400,freq)';
else 
    data = tmp.data;
end

var =  (1/400)*sum((data-mean(data,2)).^2,2);
[~,idx] = sort(var,'descend');
data = data(idx(1:numChannels),:);

wlen = 64;
hop = wlen/4;
nfft = 128;
win = hann(wlen);

if strcmp(field, "label")
    [~,name] = fileparts(matFile); % extract name of file without file ext
    name_parts = split(name, "_"); % split up name at underscores
    % Assigns category for data to relevant part of file name
    data = categorical(name_parts(3), classes);
elseif strcmp(field, "stft_real")
    %data = normalize(real(stft(data',400,'Window',hann(64))));
    %realData = normalize(real(stft(data',400,'Window',hann(64))));
    %imagData = normalize(imag(stft(data',400,'Window',hann(64))));
    S = arrayfun(@(row_idx) stft(data(row_idx,:),win,hop,nfft,400),(1:size(data,1)).','uni',false);
    S = cat(3,S{:});
    realData = normalize(real(S));
    imagData = normalize(imag(S));
    data = cat(3,realData,imagData);
elseif strcmp(field, "stft_imag")
    data = normalize(imag(stft(data',400,'Window',hann(64))));
elseif strcmp(field, "stft")
    tmpStft = stft(data',400);
    data = {real(tmpStft), imag(tmpStft)};
end

end