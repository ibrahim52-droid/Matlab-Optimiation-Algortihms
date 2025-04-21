%2024PGCSCS17
function solve_large_gap_pso()
    pso_results = [];
    gap12_convergences_pso = {};

    for g = 1:12
        filename = sprintf('./gap dataset files/gap%d.txt', g);
        fid = fopen(filename, 'r');
        if fid == -1
            error('Error opening file %s.', filename);
        end

        num_problems = fscanf(fid, '%d', 1);
        fprintf('\n%s\n', filename(1:end-4));

        for p = 1:num_problems
            m = fscanf(fid, '%d', 1);
            n = fscanf(fid, '%d', 1);
            c = fscanf(fid, '%d', [n, m])';
            r = fscanf(fid, '%d', [n, m])';
            b = fscanf(fid, '%d', [m, 1]);

            [x_matrix, convergence] = solve_gap_pso(m, n, c, r, b);
            objective_value = sum(sum(c .* x_matrix));
            pso_results(end+1, :) = [g + p/10, objective_value];

            fprintf('c%d-%d  %d\n', m*100 + n, p, round(objective_value));

            if g == 12
                gap12_convergences_pso{end+1} = convergence;
            end
        end

        fclose(fid);
    end

    % Plot convergence curves for GAP12
    load('binary_ga.mat', 'gap12_convergences_binary'); % binary coded GA
    load('real_ga.mat', 'gap12_convergences_real');     % real-coded GA

    if ~isempty(gap12_convergences_pso)
        figure;
        hold on;
        for i = 1:length(gap12_convergences_binary)
            plot(gap12_convergences_binary{i}, 'r--', 'LineWidth', 1.2);
            plot(gap12_convergences_real{i}, 'b-.', 'LineWidth', 1.2);
            plot(gap12_convergences_pso{i}, 'g-', 'LineWidth', 1.5);
        end
        title('Convergence Comparison for GAP12 Instances');
        xlabel('Iterations');
        ylabel('Best Fitness');
        legend({'Binary GA', 'Real-coded GA', 'PSO'});
        grid on;
        hold off;
    end

    save('pso_results.mat', 'pso_results', 'gap12_convergences_pso');
end
function [x_matrix, convergence] = solve_gap_pso(m, n, c, r, b)
    pop_size = 50;
    max_iter = 300;

    dim = m * n;
    lb = zeros(1, dim);
    ub = ones(1, dim);

    w = 0.7; c1 = 1.5; c2 = 1.5;

    positions = rand(pop_size, dim); % real-coded
    velocities = zeros(pop_size, dim);
    pbest = positions;
    pbest_val = arrayfun(@(i) fitnessFcn(positions(i, :)), 1:pop_size);
    [gbest_val, gbest_idx] = max(pbest_val);
    gbest = pbest(gbest_idx, :);
    convergence = zeros(1, max_iter);

    for iter = 1:max_iter
        for i = 1:pop_size
            velocities(i, :) = w * velocities(i, :) + ...
                c1 * rand * (pbest(i, :) - positions(i, :)) + ...
                c2 * rand * (gbest - positions(i, :));

            positions(i, :) = positions(i, :) + velocities(i, :);
            positions(i, :) = min(max(positions(i, :), lb), ub);

            % Enforce feasibility and re-evaluate
            corrected = enforce_feasibility(positions(i, :), m, n);
            val = fitnessFcn(corrected);

            if val > pbest_val(i)
                pbest(i, :) = positions(i, :);
                pbest_val(i) = val;
                if val > gbest_val
                    gbest = positions(i, :);
                    gbest_val = val;
                end
            end
        end
        convergence(iter) = gbest_val;
    end

    x_matrix = reshape(enforce_feasibility(gbest, m, n), m, n);

    function fval = fitnessFcn(x)
        x_mat = reshape(enforce_feasibility(x, m, n), [m, n]);
        cost = sum(sum(c .* x_mat));
        cap_violation = sum(max(sum(x_mat .* r, 2) - b, 0));
        assign_violation = sum(abs(sum(x_mat, 1) - 1));
        penalty = 1e6 * (cap_violation + assign_violation);
        fval = cost - penalty;
    end
end

function x_corrected = enforce_feasibility(x, m, n)
    x_mat = reshape(x, [m, n]);
    for j = 1:n
        [~, idx] = max(x_mat(:, j));
        x_mat(:, j) = 0;
        x_mat(idx, j) = 1;
    end
    x_corrected = reshape(x_mat, [1, m * n]);
end
