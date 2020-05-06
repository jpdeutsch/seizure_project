%function [dsTrain, dsTest] = datastores_detection_noTestData(trainPaths,...
%    testPaths,trainLabels,testLabels)
function [dsTrain, dsTest] = datastores_detection_noTestData(trainPaths,...
    testPaths,stftScenario,numChannels,classes,trainLabels,testLabels)

dsTrain = imageDatastore(trainPaths,'FileExtensions','.mat','ReadFcn', @(f)...
    loadData(f,classes,"stft_real",stftScenario,numChannels),...
    'Labels',categorical(trainLabels));
    
dsTest = imageDatastore(testPaths,'FileExtensions','.mat','ReadFcn', @(f)...
    loadData(f,classes,"stft_real",stftScenario,numChannels),...
    'Labels',categorical(testLabels));
  

%{
dsTrain = imageDatastore(trainPaths,'FileExtensions','.mat','ReadFcn', @(f)...
    loadD(f),'Labels',categorical(trainLabels));
dsTest = imageDatastore(testPaths,'FileExtensions','.mat','ReadFcn',@(f)...
    loadD(f),'Labels',categorical(testLabels));
%}


end


function data = loadD(matFile)
tmp = load(matFile);
data = tmp.data;
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
if freq ~= 200
    data = resample(tmp.data',200,freq)';
else 
    data = tmp.data;
end

% Calculate variance of each channel and only take the most variant
var =  (1/400)*sum((data-mean(data,2)).^2,2);
[~,idx] = sort(var,'descend');
data = data(idx(1:numChannels),:);

if stftScenario == 1
    wlen = 64;
    hop = wlen/4;
    nfft = 128;
    win = hann(wlen);
elseif stftScenario == 2
    wlen = 128;
    hop = wlen/4;
    nfft = 128;
    win = hann(wlen);
elseif stftScenario == 3
    wlen = 128;
    hop = wlen/4;
    nfft = 256;
    win = hann(wlen);
else
    wlen = 16;
    hop = wlen/4;
    nfft = 256;
    win = hann(wlen);
end


if strcmp(field, "label")
    [~,name] = fileparts(matFile); % extract name of file without file ext
    name_parts = split(name, "_"); % split up name at underscores
    % Assigns category for data to relevant part of file name
    data = categorical(name_parts(3), classes);
elseif strcmp(field, "stft_real")
    S = arrayfun(@(row_idx) stft(data(row_idx,:),win,hop,nfft,400),...
        (1:size(data,1)).','uni',false);
    S = cat(3,S{:}); % change stft results into "stacked" image
    realData = normalize(real(S)); % real parts of stft
    imagData = normalize(imag(S)); % imaginary parts of stft
    data = cat(3,realData,imagData);
elseif strcmp(field, "stft_imag")
    data = normalize(imag(stft(data',400,'Window',hann(64))));
elseif strcmp(field, "stft")
    tmpStft = stft(data',400);
    data = {real(tmpStft), imag(tmpStft)};
end

end