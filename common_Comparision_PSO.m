% 2024PGCSCS17

% Load results
load('ilp_results.mat');        % loads: ilp_results, instance_labels
load('approx_results.mat');     % loads: approx_results
load('ga_results.mat');         % loads: ga_results (Binary GA)
load('real_ga.mat');            % loads: ga_results (Real-coded GA)
load('pso_results.mat');        % loads: pso_results

% Rename GA variables to avoid conflict
binary_ga_results = ga_results;
real_ga_results = ga_results;

% Sort pso_results to match instance_labels order
[~, sort_idx] = sort(pso_results(:, 1));
sorted_pso_results = pso_results(sort_idx, 2);

% Combine objective values
combined_values = [ ...
    ilp_results(:,2), ...
    approx_results(:,2), ...
    binary_ga_results(:,2), ...
    real_ga_results(:,2), ...
    sorted_pso_results ...
];

% Plotting
figure;
bar(combined_values);
legend({'ILP (Optimal)', 'Approximation', 'Binary GA', 'Real-Coded GA', 'PSO'}, 'Location', 'northwest');
xticks(1:length(instance_labels));
xticklabels(instance_labels);
xtickangle(45);
xlabel('Problem Instance');
ylabel('Objective Value');
title('Comparison of GAP Solvers');
grid on;
