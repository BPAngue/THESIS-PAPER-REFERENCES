close all; clear; clc; warning('off', 'all');

addpath('MOPSO\');
addpath('MOGWO\');
addpath('MOMVO\');
addpath('MOSSA\');
addpath('MOSMA\');
addpath('Subfunctions\');
addpath(genpath('MM_testfunctions/'));

results_dir='./多目标隧道';
if ~exist('./多目标隧道', 'dir')
    mkdir('./多目标隧道');
end

% 测试问题循环，一共24个
for TestProblem = 1

    fprintf('正在处理: TP%d\n', TestProblem);

    % 创建保存评价指标，帕累托解，帕累托前沿的文件夹
    problem_dir = sprintf('%s/评价指标', results_dir);
    problem_dir1 = sprintf('%s/帕累托解', results_dir);
    problem_dir2 = sprintf('%s/帕累托前沿', results_dir);
    mkdir(problem_dir);
    mkdir(problem_dir1);
    mkdir(problem_dir2);

    % 获取函数细节
    MultiObj = Get_function_details(TestProblem);

    algorithms = {'MOGWO', 'MOMVO', 'MOSSA', 'MOSMA', 'MOPSO', 'MOIPSO'}; % 添加删除算法
    num_algorithms = length(algorithms);
    params.Np = 200; % 种群数量
    params.Nr = 200; % 存档大小
    params.maxgen = 100; % 迭代次数
    num_runs = 21 ; % 算法运行次数
    dpi = '600'; % 保存图片质量大小
    AllMetrics = {'Spacing'}; % 仅使用Spacing作为评价指标
    metrics_data = zeros(num_runs, 1, num_algorithms); % 更新指标数据结构，只存储一个指标

    % 遍历每一个算法
    for a = 1:num_algorithms
        algo = algorithms{a};
        fprintf('%s运行中\n', algo);
        for i = 1:num_runs

            disp([algo,'第',num2str(i),'次运行']);
            params.Algorithm = algo;
            tic;
            REP = feval(algorithms{a},params, MultiObj);
            execution_times(i, a) = toc;
            ps = REP.pos;
            pf = REP.pos_fit;

            % 计算Spacing指标
            SpacingValue = calculateSpacing(pf);
            metrics_data(i, :, a) = SpacingValue;

            % 绘制帕累托解和保存
            fig1 = figure;
            plot(ps(:,1), ps(:,2), 'ro');
            title(sprintf('Pareto solutions (%s)',  algo));
            box on;
            grid on;
            % axis tight;
            set(gca,'FontSize',12,'Fontname', 'Times New Roman'); 
            figname1 = sprintf('%s/%s_Run%d_ParetoSolutions.jpg', problem_dir1, algo, i);
            print(fig1, figname1, '-djpeg', ['-r', dpi]);
            close(fig1);

            % 绘制帕累托前沿和保存
            fig2 = figure;
            plot(pf(:,1), pf(:,2), 'bo');
            title(sprintf('Pareto front (%s)', algo));
            xlabel('Objective 1');
            ylabel('Objective 2');
            set(gca,'FontSize',12,'Fontname', 'Times New Roman'); 
            box on;
            grid on;
            % axis tight;
            figname2 = sprintf('%s/%s_Run%d_ParetoFront.jpg', problem_dir2, algo, i);
            print(fig2, figname2, '-djpeg', ['-r', dpi]);
            close(fig2);
        end
    end

    % 计算并保存评价指标均值和标准差
    stat_names = {'Mean', 'Std'};
    metric_values = squeeze(metrics_data(:,1,:));
    stats = zeros(2, num_algorithms);
    stats(1, :) = mean(metric_values, 1);
    stats(2, :) = std(metric_values, 0, 1);
    filename = sprintf('%s/Spacing_MeanStd.xlsx', problem_dir);
    stats_table = array2table(stats, 'VariableNames', algorithms, 'RowNames', stat_names);
    writetable(stats_table, filename, 'WriteRowNames', true);
    
    % 绘制保存每一个指标的箱线图
    fig3 = figure;
    boxplot(metric_values, 'Labels', algorithms);
    title('Boxplot');
    ylabel('Spacing');
    set(gca,'FontSize',12,'Fontname', 'Times New Roman'); 
    box on;
    grid on;
    % axis tight;
    figname3 = sprintf('%s/Spacing_Boxplot.jpg', problem_dir);
    print(fig3, figname3, '-djpeg', ['-r', dpi]);
    close(fig3);
end
