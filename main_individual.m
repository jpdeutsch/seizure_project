%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Initial Processing of Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Classes of all data
classes = ["ictal", "interictal"];

% All patients in the stu
patients = ["Dog_"+string(1:4), "Patient_"+string(1:8)];

% Base paths for seizure detection datasets of each patient
% Path for mounted drive
datasetPath = fullfile("..","all_data","Detection");
%datasetPath = "E:\School\EE5549\Detection";
ksmotePath = fullfile(datasetPath,"ksmote");
%ksmotePath = "E:\School\EE5549\Detection\ksmote";

% Path for lab computer
figurePath = fullfile("..","Figures");

patientToRun = [12];
numRuns = 3;

filterSize = [3];
numFilters = [32];
maxPool = [2];
dropout = [0.5]; 
numChannels = [8];

downsample = 200;

disp("Running Each Patient, 8 channels, no dropout, KSMOTE")

%%
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

    
    for i=1:numRuns

        fprintf("\nPatient:%1.0f, Run:%1.0f\n",p,i)

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %   Training the Network
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        inputSize = [size(trainData,1), size(trainData,2)];
        disp("Starting Training")
        
        tic
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
            evaluateResults(testLabels,pred,figurePath,scores);
        toc
    end


    fprintf("\n\n**********AVERAGES**********\n")
    avg_TP = mean(conf(1,1,:));
    avg_FP = mean(conf(1,2,:));
    avg_FN = mean(conf(2,1,:));
    avg_TN = mean(conf(2,2,:));
    fprintf("Averages: TP=%4.2f, FP=%4.2f, FN=%4.2f, TN=%4.2f\n", avg_TP, avg_FP, avg_FN, avg_TN)

    avg_sens = avg_TP/(avg_TP+avg_FN);
    avg_spec = avg_TN/(avg_TN+avg_FP);
    fprintf("Averages: Sensitivity=%.4f, Specificity=%.4f\n",avg_sens,avg_spec)

    avg_AUC = mean(AUC);
    fprintf("Average AUC: %.4f\n",avg_AUC)

end
