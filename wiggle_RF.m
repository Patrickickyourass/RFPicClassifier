clc,clear

% 读取SAC格式的地震波形数据文件
% s = readsac('D:\RFDATASET\Test_Dataset\good\SC.ABCL.20220608.159.HHR.SACi');%模板sac文件随便放一个测试看看
% f=figure;
% f.Position(4)=210;
% 
% wiggle(s.DATA1(1:500),'BR'),view(-90,90),axis("off")

datafiles=dir('C:\Users\...\Desktop\T1_RF_MechinePick\KCD01\*.sac');
if ~exist('.\test')
    mkdir('.\test');
end

for i=1:length(datafiles)
    disp(i)
    s=readsac([datafiles(i).folder,'\',datafiles(i).name]);
    pic_name=([datafiles(i).name(1:end-4),'.png']);
    if exist(['.\test\',pic_name],"file") || (s.NPTS<500) 
        disp('yes')
        continue;
    end
    % 明确使用传统的 figure 函数创建图窗
    fig = figure('Visible', 'off'); 
    wiggle(s.DATA1(1:500),'BR'),view(-90,90),axis("off")
    % plot(s.DATA1(1:500))
    saveas(fig,['.\test\',pic_name])

    close(fig)
    

end
