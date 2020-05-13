%{
%   given a patient loads in the data and preprocesses it
%   
%   inputs
%       path - path to the patients data
%       downsample - frequency to downsample to
%       numChannels - number of channels to include in the data
%}
function [trainData,trainLabels] = prepPatientTrain(path,downsample,numChannels)

% Get all ictal clips in patient directory
ictalClips = dir(fullfile(path,"*_ictal_*.mat"));

% Get all interictal clips in patient directory
interictalClips = dir(fullfile(path,"*_interictal_*.mat"));

% Get the full path for each ictal file
ictalPaths = arrayfun(@(f) fullfile(path,ictalClips(f).name),...
    [1:length(ictalClips)],'uni',false)';

% Get the full path for each interictal file
interictalPaths = arrayfun(@(f) fullfile(path,interictalClips(f).name),...
    [1:length(interictalClips)],'uni',false)';

% Load the ictal data
ictalData = zeros(numChannels,downsample,1,length(ictalPaths));
parfor i=1:length(ictalPaths)
    
    file = load(ictalPaths{i}); % load in all ictal data
    
    freq = ceil(file.freq); % frequency of original data

    % downsample the data
    if freq ~= downsample
        data = resample(file.data',downsample,freq)';
    else 
        data = file.data;
    end
    
    % Calculate variance of each channel and only take the most variant
    var =  (1/downsample)*sum((data-mean(data,2)).^2,2);
    [~,idx] = sort(var,'descend');
    ictalData(:,:,:,i) = data(idx(1:numChannels),:);
    
end
    
% Load the interictal data
interictalData = zeros(numChannels,downsample,1,length(interictalPaths));
parfor i=1:length(interictalPaths)
   
    file = load(interictalPaths{i}); % load in all interictal data
    
    freq = ceil(file.freq); % frequency of original data

    % downsample the data
    if freq ~= downsample
        data = resample(file.data',downsample,freq)';
    else 
        data = file.data;
    end
    
    % Calculate variance of each channel and only take the most variant
    var =  (1/downsample)*sum((data-mean(data,2)).^2,2);
    [~,idx] = sort(var,'descend');
    interictalData(:,:,:,i) = data(idx(1:numChannels),:);
       
end

% Put all training data into 1 4-D cell array
trainData = cat(4,ictalData,interictalData);

% Generate labels for training data
trainLabels = categorical(vertcat(repmat("ictal",[length(ictalPaths) 1]),...
    repmat("interictal",[length(interictalPaths) 1])));

end
