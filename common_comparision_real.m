%2024PGCSCS17
load('ilp_results.mat');        % loads: ilp_results, instance_labels
load('approx_results.mat');     % loads: approx_results
load('ga_results.mat');         % loads: ga_results (Binary GA)
load('real_ga.mat');            % loads: ga_results (Real-coded GA)

% Rename real GA variable to avoid overwriting
real_ga_results = ga_results;

% Combine objective values
combined_values = [ ...
    ilp_results(:,2), ...
    approx_results(:,2), ...
    ga_results(:,2), ...
    real_ga_results(:,2)
];

% Plotting
figure;
bar(combined_values);
legend({'ILP (Optimal)', 'Approximation', 'Binary GA', 'Real-Coded GA'}, 'Location', 'northwest');
xticks(1:length(instance_labels));
xticklabels(instance_labels);
xtickangle(45);
xlabel('Problem Instance');
ylabel('Objective Value');
title('Comparison of GAP Solvers');
grid on;
