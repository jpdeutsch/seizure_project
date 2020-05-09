function [conf,sens,spec,AUC] = evaluateResults(testLabels,pred,path,scores)

target = [(strcmp(testLabels',"ictal"));(strcmp(testLabels',"interictal"))];
output = [(pred=="ictal")'; (pred=="interictal")'];

[~,~,~,AUC] = perfcurve(testLabels,scores(:,1)',"ictal");
AUC

%h=figure();
%plotroc(target,output);
%axesUserData=get(gca,'userdata');
%legend(axesUserData.lines,'ictal','interictal');
%title("Patient " + scenario);
%saveas(h,strcat(path,"/ROC",num2str(scenario)),'fig');
%close(h);

%g = figure();
[c,cm,ind,per] = confusion(int8(target),int8(output));
%plotconfusion(int8(target),int8(output));
%title("Patient " + scenario);

TP = cm(1)
FP = cm(2)
FN = cm(3)
TN = cm(4)
conf = [TP FP; FN TN];
sens = TP/(TP+FN)
spec = TN/(TN+FP)

end
