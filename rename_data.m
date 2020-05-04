function rename_data(folder)
files = dir(fullfile(folder,"*.mat"));

for id = 1:length(files)
   [~,f,ext] = fileparts(files(id).name);
   splitName = split(f,"_");
   
   newName = [strjoin([[splitName(1:end-1)]',...
       num2str(str2double(splitName{end}),'%04.f')],"_"),ext];
   
   oldPath = fullfile(folder,files(id).name);
   newPath = fullfile(folder,newName);
   
   if ~strcmp(oldPath,newPath) 
       movefile(fullfile(folder,files(id).name), fullfile(folder,newName));
   end
end
end