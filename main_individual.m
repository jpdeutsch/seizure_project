%{
%   Main script to run which loads up the training data and testing data, trains
%   the network and evaluates performance
%
%   Parameters to vary
%       patientToRun - the patients to train a network for
%       numRuns - number of networks to train for each patient and average
%       filterSize - size of the convolutional filters
%       maxPool - size of the max pooling layers
%       numChannels - number of IEEG channels to use for the network
%
%       downsample - frequency to downsample the data to
%}


% Classes of all data
classes = ["ictal", "interictal"];

% All patients in the stu
patients = ["Dog_"+string(1:4), "Patient_"+string(1:8)];

% Base paths for seizure detection datasets of each patient
datasetPath = fullfile("..","all_data","Detection"); 

% Path to data which has had K-SMOTE run on it
ksmotePath = fullfile(datasetPath,"ksmote");

% Where to save figures
figurePath = fullfile("..","Figures");

% Patients to train a network for
patientToRun = [3, 5, 7];

% Number of runs for each patient
numRuns = 1;

filterSize = 4; % conv filter size
numFilters = 32; % number of conv filters
maxPool = 2; % max pool size
numChannels = 8; % number of input channels to use

downsample = 200; % freq to downsample to

disp("Running Each Patient, 8 channels, no dropout, KSMOTE")

% Loop through each patient
for p=patientToRun
    fprintf("\n**********STARTING PATIENT %1.0f**********\n",p)

    conf = zeros(2,2,numRuns);
    sens = zeros(1,numRuns);
    spec = zeros(1,numRuns);
    AUC = zeros(1,numRuns);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Preprocess the data
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    [trainData, trainLabels, testPaths, testLabels,valPaths,valLabels] = ...
        preprocess(datasetPath,ksmotePath,patients(p),numChannels,downsample,1);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Put data into datastores for network
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    [dsTest,dsVal] = datastores(valPaths,valLabels,testPaths,testLabels,numChannels);
    
    % create numRuns amount of networks for each patient
    for i=1:numRuns

        fprintf("\nPatient:%1.0f, Run:%1.0f\n",p,i)

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %   Training the Network
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        inputSize = [size(trainData,1), size(trainData,2)]; % input size to the network

        disp("Starting Training")
        
        % Build and train the network
        net = buildNetwork(trainData,trainLabels,dsVal,inputSize,filterSize,numFilters,...
            maxPool);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %   Run the network on the test data
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        disp("Classifying")

        [pred, scores] = classify(net, dsTest);


        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %   Evaluate Network Performance
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        fprintf("\nResults for Patient %1.0f, run %1.0f\n",p,i)

        [conf(:,:,i),sens(i),spec(i),AUC(i)] = ...
            evaluateResults(testLabels,pred,figurePath,scores,patients(p));

    end

    % Average out each run of the network
    fprintf("\n\n**********AVERAGES**********\n")
    avg_TP = mean(conf(1,1,:)); % True positive
    avg_FP = mean(conf(1,2,:)); % False positive
    avg_FN = mean(conf(2,1,:)); % False negative
    avg_TN = mean(conf(2,2,:)); % True negative

    fprintf("Averages: TP=%4.2f, FP=%4.2f, FN=%4.2f, TN=%4.2f\n", avg_TP, avg_FP, avg_FN, avg_TN)

    avg_sens = avg_TP/(avg_TP+avg_FN); % Sensitivity
    avg_spec = avg_TN/(avg_TN+avg_FP); % Specificity

    fprintf("Averages: Sensitivity=%.4f, Specificity=%.4f\n",avg_sens,avg_spec)

    avg_AUC = mean(AUC); % Area under ROC curve

    fprintf("Average AUC: %.4f\n",avg_AUC)

end
