function [trainPaths, trainLabels, testPaths, testLabels] = ...
    preprocess_detection_noTestData(path,patients,whichPatients)

% Fraction of data to be in training group
trainFrac = 1/2;

% Full path to each patient's data
pathPatients = fullfile(path, patients);

% General paths to all prelabeled data
ictal_paths_gen = fullfile(pathPatients, "*_ictal*.mat");
interictal_paths_gen = fullfile(pathPatients, "*_interictal*.mat");

% File objects for all prelabeled data
ictalFiles = cellfun(@(x) dir(x),ictal_paths_gen,'UniformOutput',false);
interictalFiles = cellfun(@(x) dir(x),interictal_paths_gen,'UniformOutput',false);

% Full path to all prelabeled data
ictalPaths = cellfun(@(file) arrayfun(@(struct) fullfile(struct.folder,struct.name),...
    file,'uni',false),ictalFiles,'uni',false);
interictalPaths = cellfun(@(file) arrayfun(@(struct) fullfile(struct.folder,struct.name),...
    file,'uni',false),interictalFiles,'uni',false);

% Array of the amount of each patient's ictal data segments
totalIctal = cellfun(@(c) length(c), ictalFiles,'uni',true);
% Array of the amount of each patient's interictal data segments
totalInterictal = cellfun(@(c) length(c), interictalFiles,...
    'uni',true);
interToIctal = totalInterictal./totalIctal;

% Arrays of the amount of each patient for training data
trainIctal = ceil(totalIctal.*trainFrac);
%trainInterictal = ceil(totalInterictal.*trainFrac);
%trainInterictal = floor(totalIctal.*1.4);
trainInterictal = totalIctal.*(floor(min(interToIctal,2)));

% TODO: hardcoded for just dogs right now, use startPatient and endPatient
% to preallocate array and speed things up
% Total ictal and interictal data for all patients
%trainingPath = strings(1,sum(trainIctal(1:4))+sum(trainInterictal(1:4)));
%validationPath = strings(1,sum(totalIctal(1:4)-trainIctal(1:4))+...
%    sum(totalInterictal(1:4)-trainInterictal(1:4)));
trainPaths = []; testPaths = [];

for i=whichPatients
    trainPaths = [trainPaths,...
        [ictalPaths{i}(1:trainIctal(i))]',...
        [interictalPaths{i}(1:trainInterictal(i))]'];
    testPaths = [testPaths,...
        [ictalPaths{i}(trainIctal(i)+1:totalIctal(i))]',...
        [interictalPaths{i}(trainInterictal(i)+1:totalInterictal(i))]'];
end
%{
for i=whichPatients
    trainPaths = [trainPaths,...
        [ictalPaths{i}(1:trainIctal(i))]',...
        [interictalPaths{i}(1:trainInterictal(i))]'];
    testPaths = [testPaths,...
        [ictalPaths{i}(trainIctal(i)+1:totalIctal(i))]',...
        [interictalPaths{i}(trainInterictal(i)+1:totalInterictal(i))]'];
end
%}

% Creates arrays of the correct labels for each train and test data set
trainLabels = strings(1,length(trainPaths));
trainLabels(contains(trainPaths,"_ictal")) = "ictal";
trainLabels(contains(trainPaths,"_interictal")) = "interictal";

testLabels = strings(1,length(testPaths));
testLabels(contains(testPaths,"_ictal")) = "ictal";
testLabels(contains(testPaths,"_interictal")) = "interictal";

end
