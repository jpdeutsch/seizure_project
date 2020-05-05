function evaluateResults(testLabels,pred,scenario,path)
%{
targetLabels = strings(1,length(dsTest.Files));
i=1;
while hasdata(dsTest)
    targetLabels(i) = string(read(dsTest));
    i = i+1;
end
%}

target = [(strcmp(testLabels,"ictal"));(strcmp(testLabels,"interictal"))];
output = [(pred=="ictal")'; (pred=="interictal")'];

h=figure();
plotroc(target,output);
axesUserData=get(gca,'userdata');
legend(axesUserData.lines,'ictal','interictal');
title("Scenario " + scenario);
saveas(h,strcat(path,"\ROC",num2str(scenario)),'fig');
close(h);

h=figure();
[c,cm,ind,per] = confusion(int8(target),int8(output));
plotconfusion(int8(target),int8(output));
title("Scenario " + scenario);
saveas(h,strcat(path,"\Confusion",num2str(scenario)),'fig');
close(h);

end
