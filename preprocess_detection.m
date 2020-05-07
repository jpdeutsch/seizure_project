function [trainPaths, trainLabels, testPaths, testLabels, valPaths, valLabels] = ...
    preprocess_detection(path,patients,whichPatients,numChannels)
% Fraction of data to be in training group
trainFrac = 1/2;
downsample = 200;

trainPaths = {}; testPaths = {};
trainLabels = {}; testLabels = {};
valPaths = {}; valLabels = {};

for i=whichPatients
    patientPath = fullfile(path,patients(i));
    [ictalPaths,interictalPaths] = prepPatient(patientPath);
    [testIctal,testInterictal] = prepTestData(path,patientPath,patients(i));
    
%{    
    ictalData = cell(length(ictalPaths),1);
    parfor i=1:length(ictalPaths)
    
        file = load(ictalPaths{i});
    
        freq = ceil(file.freq);
        if freq ~= downsample
            data = resample(file.data',downsample,freq)';
        else 
            data = file.data;
        end
    
        % Calculate variance of each channel and only take the most variant
        var =  (1/200)*sum((data-mean(data,2)).^2,2);
        [~,idx] = sort(var,'descend');
        ictalData{i} = data(idx(1:numChannels),:);
    
    end
    
    interictalData = cell(length(interictalPaths),1); 
    parfor i=1:length(interictalPaths)
    
        file = load(interictalPaths{i});
    
        freq = ceil(file.freq);
        if freq ~= downsample
            data = resample(file.data',downsample,freq)';
        else 
            data = file.data;
        end
    
        % Calculate variance of each channel and only take the most variant
        var =  (1/200)*sum((data-mean(data,2)).^2,2);
        [~,idx] = sort(var,'descend');
        interictalData{i} = data(idx(1:numChannels),:);
       
    end
    
    ictalTestData = cell(length(testIctal),1);
    parfor i=1:length(testIctal)
    
        file = load(testIctal{i});
    
        freq = ceil(file.freq);
        if freq ~= downsample
            data = resample(file.data',downsample,freq)';
        else 
            data = file.data;
        end
    
        % Calculate variance of each channel and only take the most variant
        var =  (1/200)*sum((data-mean(data,2)).^2,2);
        [~,idx] = sort(var,'descend');
        ictalTestData{i} = data(idx(1:numChannels),:);
    
    end
    
    interictalTestData = cell(length(testInterictal),1);
    parfor i=1:length(testInterictal)
    
        file = load(testInterictal{i});
    
        freq = ceil(file.freq);
        if freq ~= downsample
            data = resample(file.data',downsample,freq)';
        else 
            data = file.data;
        end
    
        % Calculate variance of each channel and only take the most variant
        var =  (1/downsample)*sum((data-mean(data,2)).^2,2);
        [~,idx] = sort(var,'descend');
        interictalTestData{i} = data(idx(1:numChannels),:);
    
    end
    %}
    
    %trainPaths = vertcat(trainPaths,ictalData,interictalData);
    trainPaths = vertcat(trainPaths,ictalPaths,interictalPaths);
    trainLabels = vertcat(trainLabels,repmat("ictal",[length(ictalPaths) 1]),...
        repmat("interictal",[length(interictalPaths) 1]));
    
    valSplitIctal = randperm(length(testIctal));
    valSplitInterictal = randperm(length(testInterictal));
    %valSplitIctal = randperm(length(ictalTestData));
    %valSplitInterictal = randperm(length(interictalTestData));
    valIctal = testIctal(valSplitIctal(1:floor(0.2*length(testIctal))));
    valInterictal = testInterictal(valSplitInterictal(1:floor(0.2*length(testInterictal))));
    valPaths = vertcat(valIctal,valInterictal);
    valLabels = vertcat(repmat("ictal",[length(valIctal),1]),...
        repmat("interictal",[length(valInterictal),1]));
    
    
    %ictalTestData = ictalTestData(valSplitIctal(floor(0.2*length(ictalTestData))+1:end));
    %interictalTestData = interictalTestData(valSplitInterictal(floor(0.2*length(interictalTestData))+1:end));
    testIctal = testIctal(valSplitIctal(floor(0.2*length(testIctal))+1:end));
    testInterictal = testInterictal(valSplitInterictal(floor(0.2*length(testInterictal))+1:end));

    
    %testPaths = vertcat(testPaths,ictalTestData,interictalTestData);
    testPaths = vertcat(testPaths,testIctal,testInterictal);
    testLabels = vertcat(testLabels,repmat("ictal",[length(testIctal),1]),...
        repmat("interictal",[length(testInterictal) 1]));
end

end

function [ictalPaths,interictalPaths] = prepPatient(path)
ictalClips = dir(fullfile(path,"*_ictal_*.mat"));
interictalClips = dir(fullfile(path,"*_interictal_*.mat"));
testClips = dir(fullfile(path,"*_test_*.mat"));

ictalPaths = arrayfun(@(f) fullfile(path,ictalClips(f).name),...
    [1:length(ictalClips)],'uni',false)';

interictalPaths = arrayfun(@(f) fullfile(path,interictalClips(f).name),...
    [1:length(interictalClips)],'uni',false)';

%oversample ictal signals
over_idx = randperm(length(ictalPaths),floor(0.2*length(ictalPaths)));
ictalPaths = vertcat(ictalPaths,ictalPaths(over_idx));

%undersample interictal signals
under_idx = randperm(length(interictalPaths),floor(0.7*length(interictalPaths)));
interictalPaths = interictalPaths(under_idx);

end

%{
% Manipulates test data labels to be used in testing the neural network.
%
% Parameters:
%   dataPath - Path to where the test .csv data is located
% Output:
%   testData - Table of modified test data to be used in network testing
%}
function [testIctal, testInterictal] = prepTestData(dataPath,patientPath,patient)
% Load data from .csv into Matlab
testData = readtable(fullfile(dataPath, "SzDetectionAnswerKey.csv"));

patientTests = testData(contains(testData.clip,patient),:);
[~, name, ~] = arrayfun(@(f) fileparts(patientTests.clip{f}),[1:size(patientTests)],...
    'uni',false);
splitName = split(name',"_");
patientTests.newName = arrayfun(@(f) [strjoin([splitName(f,1:end-1),...
    num2str(str2double(splitName{f,end}),'%04.f')],"_"), '.mat'],...
    (1:size(patientTests)),'uni',false)';


testIctal = fullfile(patientPath,...
    patientTests.newName(patientTests.seizure==1));
testInterictal = fullfile(patientPath,...
    patientTests.newName(patientTests.seizure==-1));

end
