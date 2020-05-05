function c = evaluateResults(testLabels,pred,scenario,path)

target = [(strcmp(testLabels,"ictal"));(strcmp(testLabels,"interictal"))];
output = [(pred=="ictal")'; (pred=="interictal")'];

h=figure();
plotroc(target,output);
axesUserData=get(gca,'userdata');
legend(axesUserData.lines,'ictal','interictal');
title("Scenario " + scenario);
saveas(h,strcat(path,"/ROC",num2str(scenario)),'fig');
close(h);

[conf.c,conf.cm,conf.ind,conf.per] = confusion(int8(target),int8(output));
c = conf.c;

save(strcat(path,"/confusion",num2str(scenario),".mat"),'-struct','conf');
save(strcat("../all_data/confusion",num2str(scenario),".mat"),'-struct','conf');
end
