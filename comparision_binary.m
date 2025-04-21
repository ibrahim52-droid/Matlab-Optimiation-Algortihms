load('ilp_results.mat');        % loads: ilp_results, instance_labels
load('approx_results.mat');     % loads: approx_results
load('binary_GA.mat');         % loads: binary_GA

% Combine objective values
combined_values = [ ...
    ilp_results(:,2), ...
    approx_results(:,2), ...
    ga_results(:,2)
];

% Plotting
figure;
bar(combined_values);
legend({'ILP (Optimal)', 'Approximation', 'Binary GA'}, 'Location', 'northwest');
xticks(1:length(instance_labels));
xticklabels(instance_labels);
xtickangle(45);
xlabel('Problem Instance');
ylabel('Objective Value');
title('Comparison of GAP Solvers');
grid on;
