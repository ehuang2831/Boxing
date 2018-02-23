% D = readtable('C:\Users\cedric.fraces\Dropbox (Personal)\Stanford\Classes\CS-230 Deep Learning\Project\tf-openpose\Poses Output\Boxing\Fight_Boxing3100.csv');
D = readtable('pose_test.csv');

connections = [0,1;1,2;2,3;0,4;4,5;5,6;0,7;7,8;...
    8,9;9,10;8,11;11,12;12,13;8,14;14,15;15,16];
connection = connections+1;

body_parts = {'RHi','RL','RF','LHi','LL','LF',...
    'B','T','N','H','LS','LA','LH',...
    'RS','RA','RH'};

t = [diff(D.Var1);-1];
j=1;
for k=find(t<0)'
    d = D(j:k,:);
    j=k+1;
    clf
    hold on
    text(d.x(connection(1,1)),d.y(connection(1,1)),d.z(connection(i,2)),'P')
    for i=1:size(connections,1)
%         plot3([d.x(connection(i,:))],[d.y(connection(i,:))],[d.z(connection(i,:))],'-', 'LineWidth',2)
        plot3(d.x(connection(i,:)),d.y(connection(i,:)),d.z(connection(i,:)),'-o', 'LineWidth',3)
        text(d.x(connection(i,2)),d.y(connection(i,2)),d.z(connection(i,2)),char(body_parts{i}))
%         grid on
%         set(gca,'xticklabel',[])
%         set(gca,'yticklabel',[])
%         set(gca,'zticklabel',[])
        view(29,6)
%         pause
    end
    axis off
    hold off
    pause
end