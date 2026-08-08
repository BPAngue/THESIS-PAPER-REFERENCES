clc; clear ;close all

% 初始化保存数组
rPSP_ALL = [] ;rHV_ALL = [] ;IGDx_ALL = [] ;IGDf_ALL = [] ;
rPSP_Sign = [] ;rHV_Sign = [] ;IGDx_Sign = [] ;IGDf_Sign = [] ;

% 导入数据
results_dir='./MO_CEC2020结果';
for TestProblem = 1:20

    problem_dir = sprintf('%s/TP%d评价指标', results_dir, TestProblem);
    AllMetrics = {'rPSP', 'rHV', 'IGDx', 'IGDf'}; % 评价指标初始化

    for m = 1:length(AllMetrics)

        filename = sprintf('%s/%s_均值和标准差.xlsx',problem_dir, AllMetrics{m});
        filename1 = sprintf('%s/%s_威尔森符号检验.xlsx',problem_dir, AllMetrics{m});
        temp = readtable(filename);
        temp1 = readcell(filename1);

        if m == 1
            rPSP_ALL = [rPSP_ALL ; temp] ;
            rPSP_Sign = [rPSP_Sign ; temp1] ;
        elseif m == 2
            rHV_ALL = [rHV_ALL ; temp] ;
            rHV_Sign = [rHV_Sign ; temp1] ;
        elseif m == 3
            IGDx_ALL = [IGDx_ALL ; temp] ;
            IGDx_Sign = [IGDx_Sign ; temp1] ;
        else
            IGDf_ALL = [IGDf_ALL ; temp] ;
            IGDf_Sign = [IGDf_Sign ; temp1] ;
        end

    end

end

results ='./结果统计';
if ~exist('.结果统计', 'dir')
    mkdir('./结果统计');
end

% 统计均值标准差
for m = 1:length(AllMetrics)
    filename3 = sprintf('./%s/%s_指标统计.xlsx',results, AllMetrics{m});
        if m == 1
            writetable(rPSP_ALL, filename3);
        elseif m == 2
            writetable(rHV_ALL, filename3);
        elseif m == 3
            writetable(IGDx_ALL, filename3);
        else
            writetable(IGDf_ALL, filename3);
        end

end

% 统计符号检验
for m = 1:length(AllMetrics)
    filename4 = sprintf('./%s/%s_符号检验统计.xlsx',results, AllMetrics{m});
        if m == 1
            writecell(rPSP_Sign, filename4);
        elseif m == 2
            writecell(rHV_Sign, filename4);
        elseif m == 3
            writecell(IGDx_Sign, filename4);
        else
            writecell(IGDf_Sign, filename4);
        end

end

% 计算保存弗里德曼检验结果，包括检测差异性的Friedman-p-value
algorithms = {'MOGWO', 'MOMVO', 'MOSSA', 'MOSMA', 'MOPSO', 'MOIPSO', 'Friedman-p-value'}; % 添加删除算法
stat_names = {'rPSP', 'rHV', 'IGDx', 'IGDf'}; % 评价指标初始化
Ranking_matrix_all = [] ;
for m = 1:length(AllMetrics)
    filename5 = sprintf('./%s/%s_指标统计.xlsx',results, AllMetrics{m});
    data = xlsread (filename5) ;
    [p,~,Frk] = friedman(data,1,'off'); % 计算Friedman值
    Ranking_matrix = [Frk.meanranks, p];
    Ranking_matrix_all = [Ranking_matrix_all; Ranking_matrix];
end
filename6 = sprintf('./%s/弗里德曼检验.xlsx',results);
stats_table = array2table(Ranking_matrix_all, 'VariableNames', algorithms, 'RowNames', stat_names);
writetable(stats_table, filename6, 'WriteRowNames', true);
