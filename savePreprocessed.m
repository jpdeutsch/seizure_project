function [dsIctal, dsInterictal] = savePreprocessed(origPath,newPath,patients)

origPatientPaths = fullfile(origPath,patients);
newPatientPaths = fullfile(newPath,patients);

ictal_paths = fullfile(origPatientPaths, "*_ictal*.mat");
interictal_paths = fullfile(origPatientPaths, "*_interictal*.mat");

ic = ictal_paths;
in = interictal_paths;

dsIctal = imageDatastore(ic,'FileExtensions','.mat','ReadFcn', @(f)...
    loadData(f,"stft",2,4),...
    'LabelSource', 'foldernames');

dsInterictal = imageDatastore(in,'FileExtensions','.mat','ReadFcn', @(f)...
    loadData(f,"stft",2,4),...
    'LabelSource', 'foldernames');

for i=1:length(dsInterictal.Files)
[~,n,e] = fileparts(dsInterictal.Files{i});
fullNewPath = fullfile(newPath,string(dsInterictal.Labels(i)),strcat(n,e));
data = read(dsInterictal);
save(fullNewPath,'data');
end

end

function data = loadData(matFile,field,stftScenario,numChannels)
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
    wlen = 64;
    hop = wlen/4;
    nfft = 256;
    win = hann(wlen);
end


if strcmp(field, "stft")
    S = arrayfun(@(row_idx) stft(data(row_idx,:),win,hop,nfft,400),...
        (1:size(data,1)).','uni',false);
    %S = cat(3,S{:}); % change stft results into "stacked" image
    realData = arrayfun(@(c) normalize(real(S{c})),1:4,'uni',false);
    %realData = normalize(real(S)); % real parts of stft
    %imagData = normalize(imag(S)); % imaginary parts of stft
    imagData = arrayfun(@(c) normalize(real(S{c})),1:4,'uni',false);
    data = cat(3,realData{1},imagData{1},realData{2},imagData{2},...
        realData{3},imagData{3},realData{4},imagData{4});
end

end