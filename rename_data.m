%{
%   Renames the original data from kaggle so that when loaded by Matlab the segments
%   are in the correct numerical order
%
%   inputs
%       folder - the folder of the patient to rename all files
%}
function rename_data(folder)

% load all data files in the input folder
files = dir(fullfile(folder,"*.mat"));

% loop through each file
for id = 1:length(files)

    % Split name into relevant parts
    [~,f,ext] = fileparts(files(id).name);
    splitName = split(f,"_");
   
    % Change number to have leading zeros
    newName = [strjoin([[splitName(1:end-1)]',...
        num2str(str2double(splitName{end}),'%04.f')],"_"),ext];
   
    oldPath = fullfile(folder,files(id).name);
    newPath = fullfile(folder,newName);
   
    % Save the file with the new name
    if ~strcmp(oldPath,newPath) 
        movefile(fullfile(folder,files(id).name), fullfile(folder,newName));
    end
end
end
