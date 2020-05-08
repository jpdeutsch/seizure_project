function ksmote(origPath,ksmotePath,patient,numChannels,downsample)

% Full path to patient data
patientPath = fullfile(origPath,patient);

% Paths to ictal data and interictal data for this patient
[trainData,trainLabels] = prepPatientTrain(patientPath,downsample,numChannels);


numSamples = size(trainData,4);
data_reshaped = zeros(numSamples,downsample*numChannels);
for i=1:numSamples
    for c=1:numChannels
        start = ((c-1)*downsample)+1;
        stop = c*downsample;
        data_reshaped(i,start:stop) = trainData(c,:,1,i);
    end
end

patientData.data = data_reshaped;
patientData.labels = double(trainLabels=="ictal");

save(fullfile(ksmotePath,strcat(patient,".mat")),'-struct','patientData');

end
