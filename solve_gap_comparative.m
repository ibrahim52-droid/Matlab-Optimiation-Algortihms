function solve_gap_comparative()
    optimal_values = [];
    approx_values = [];
    instance_labels = {};

    output_file = fopen('gap_comparison_output.txt', 'w');
    if output_file == -1
        error('Could not open output file.');
    end

    for g = 1:12
        filename = sprintf('./gap dataset files/gap%d.txt', g);
        fid = fopen(filename, 'r');
        if fid == -1
            error('Error opening file %s.', filename);
        end

        num_problems = fscanf(fid, '%d', 1);
        fprintf('\n%s\n', filename(1:end-4));
        fprintf(output_file, '\n%s\n', filename(1:end-4));

        for p = 1:num_problems
            m = fscanf(fid, '%d', 1);  % servers
            n = fscanf(fid, '%d', 1);  % users

            c = fscanf(fid, '%d', [n, m])';  % cost: m x n
            r = fscanf(fid, '%d', [n, m])';  % resource: m x n
            b = fscanf(fid, '%d', [m, 1]);   % capacities

            % Optimal via ILP
            x_opt = solve_gap_ilp(m, n, c, r, b);
            obj_opt = sum(sum(c .* x_opt));

            % Approximate via greedy
            x_approx = solve_gap_greedy(m, n, c, r, b);
            obj_approx = sum(sum(c .* x_approx));

            % Record
            label = sprintf('gap%d-%d', g, p);
            fprintf('Instance: %s | Optimal: %d | Approx: %d\n', label, obj_opt, obj_approx);
            fprintf(output_file, 'Instance: %s | Optimal: %d | Approx: %d\n', label, obj_opt, obj_approx);

            optimal_values(end+1) = obj_opt;
            approx_values(end+1) = obj_approx;
            instance_labels{end+1} = label;
        end

        fclose(fid);
    end

    fclose(output_file);

    % Plot comparison
    figure;
    plot(optimal_values, '-o', 'LineWidth', 2); hold on;
    plot(approx_values, '-x', 'LineWidth', 2);
    title('GAP: ILP vs Approximation');
    xlabel('Instance');
    ylabel('Objective Value');
    legend('Optimal (ILP)', 'Greedy Approximation', 'Location', 'Best');
    xticks(1:length(instance_labels));
    xticklabels(instance_labels);
    xtickangle(45);
    grid on;
end

% ILP Solver
function x_matrix = solve_gap_ilp(m, n, c, r, b)
    num_vars = m * n;
    f = -reshape(c, [num_vars, 1]);
    intcon = 1:num_vars;
    lb = zeros(num_vars, 1);
    ub = ones(num_vars, 1);

    Aeq = zeros(n, num_vars);
    beq = ones(n, 1);
    for j = 1:n
        for i = 1:m
            idx = (j - 1) * m + i;
            Aeq(j, idx) = 1;
        end
    end

    A = zeros(m, num_vars);
    for i = 1:m
        for j = 1:n
            idx = (j - 1) * m + i;
            A(i, idx) = r(i, j);
        end
    end

    options = optimoptions('intlinprog', 'Display', 'off');
    [x, ~, exitflag] = intlinprog(f, intcon, A, b, Aeq, beq, lb, ub, options);

    if exitflag <= 0
        warning('No feasible solution.');
        x_matrix = zeros(m, n);
    else
        x_matrix = reshape(round(x), [m, n]);
    end
end

% Greedy Approximation Solver
function x_matrix = solve_gap_greedy(m, n, c, r, b)
    x_matrix = zeros(m, n);
    remaining_b = b;

    % For each user, assign to server with least cost if feasible
    for j = 1:n
        best_cost = Inf;
        best_i = -1;
        for i = 1:m
            if r(i, j) <= remaining_b(i) && c(i, j) < best_cost
                best_cost = c(i, j);
                best_i = i;
            end
        end
        if best_i ~= -1
            x_matrix(best_i, j) = 1;
            remaining_b(best_i) = remaining_b(best_i) - r(best_i, j);
        end
    end
end
