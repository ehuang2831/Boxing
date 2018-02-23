% function PrepareData
clear
PoseFolder = 'C:\Users\cedric.fraces\Dropbox (Personal)\Stanford\Classes\CS-230 Deep Learning\Project\tf-openpose\Poses Output\';
% PoseFolder = 'C:\Users\cfraces\Dropbox\Stanford\Classes\CS-230 Deep Learning\Project\tf-openpose\Poses Output\';
listFiles = struct2table(dir([PoseFolder,'**/*.csv']));


connections = [0,1;1,2;2,3;0,4;4,5;5,6;0,7;7,8;...
    8,9;9,10;8,11;11,12;12,13;8,14;14,15;15,16];
connection = connections+1;

body_parts = {'Pelvis','R_Hip','R_Leg','R_Foot','L_Hip','L_Leg','L_Foot',...
    'Belly','Thorax','Neck','Head','L_Shoulder','L_Arm','L_Hand',...
    'R_Shoulder','R_Arm','R_Hand'};

n=1e6;
data = zeros(n,17*3);
target = -ones(n,1);
% Feature Names
feature_names = [strcat(body_parts,'_x'), strcat(body_parts,'_y'), strcat(body_parts,'_z')]';
frame = 1;

for i=1: size(listFiles,1)
    if rem(i,100)==0
        disp(['Processing file ',char(listFiles.name(i)),'......................']);
    end
    D = readtable([char(listFiles.folder(i)),'\',char(listFiles.name(i))]);

    t = [diff(D.Var1);-1];
    pointer=1;
    for k=find(t<0)'
        d = D(pointer:k,:);
        pointer = k + 1;
        data(frame,:) = [d.x',d.y',d.z'];
        % is boxing
        target(frame) = contains(listFiles.name(i),'boxing','IgnoreCase',true);
        frame = frame + 1;
    end
end

target(target==-1)=[];
data(sum(abs(data),2)==0,:)=[];

n_boxers = sum(target);
n_non_boxers = size(target,1)-n_boxers;

%shuffle data
idsh = randperm(size(target,1));
target = target(idsh);
data = data(idsh,:);
% Save raw data
disp(['Saving ',num2str(n_boxers),' frames of boxers'])
disp(['Saving ',num2str(n_non_boxers),' frames of non boxers'])
csvwrite('target.csv',target);
csvwrite('data.csv',data);
writetable(cell2table(feature_names),'feature_names.csv');

% split data more evenly

idx_b = target == 1;
idx = target==0;
tmp_target = target(idx);
tmp_data = data(idx,:);
% relative size of non boxers space (*1, *2,...) boxers space size
rel_size = 2;
idx_nb = randperm(n_non_boxers,rel_size*n_boxers);
target_nb = tmp_target(idx_nb);
data_nb = tmp_data(idx_nb,:);

target2 = [target(idx_b);target_nb];
data2 = [data(idx_b,:);data_nb];

disp(['Saving ',num2str(sum(target2)),' frames of boxers'])
disp(['Saving ',num2str(size(target2,1)-sum(target2)),' frames of non boxers'])
%shuffle data
idsh = randperm(size(target2,1));
target2 = target2(idsh);
data2 = data2(idsh,:);

csvwrite('target_even.csv',target2);
csvwrite('data_even.csv',data2);
% writetable(cell2table(feature_names),'feature_names.csv');