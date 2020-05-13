%{
%   Transforms the training data into a format which can be used by the
%   python imbalanced training library and saves to disk
%
%   inputs
%       origPath - path to original data
%       ksmotePath - path to save data for K SMOTE
%       patient - patient currently doing K SMOTE on
%       numChannels - number of channels for data
%       downsample - frequency of data
%
%}
function ksmote(origPath,ksmotePath,patient,numChannels,downsample)

% Full path to patient data
patientPath = fullfile(origPath,patient);

% Paths to ictal data and interictal data for this patient
[trainData,trainLabels] = prepPatientTrain(patientPath,downsample,numChannels);

numSamples = size(trainData,4); % number of training segments

% empty array for reshaped data
data_reshaped = zeros(numSamples,downsample*numChannels);

% Reshape data into format for ksmote algorithm
tmp = reshape(trainData,[numChannels downsample numSamples]);
data_reshaped = permute(reshape(permute(tmp,[2 1 3]),...
    [1 downsample*numChannels numSamples]),[3 2 1]);

% Save data and labels to a struct
patientData.data = data_reshaped;
patientData.labels = double(trainLabels=="ictal");

% Save struct to file
save(fullfile(ksmotePath,strcat(patient,"_",string(numChannels),"chan.mat")),'-struct','patientData');

end
