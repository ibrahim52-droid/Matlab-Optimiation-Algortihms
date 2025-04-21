function solve_large_gap_ilp()
    objective_values = [];
    instance_labels = {};

    % Open output file for writing
    output_file = fopen('gap_output.txt', 'w');
    if output_file == -1
        error('Failed to open output file.');
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
            m = fscanf(fid, '%d', 1);
            n = fscanf(fid, '%d', 1);

            c = fscanf(fid, '%d', [n, m])';
            r = fscanf(fid, '%d', [n, m])';
            b = fscanf(fid, '%d', [m, 1]);

            x_matrix = solve_gap_ilp(m, n, c, r, b);
            objective_value = sum(sum(c .* x_matrix));

            label = sprintf('c%d-%d', m * 100 + n, p);
            output_line = sprintf('%s  %d\n', label, round(objective_value));

            % Print to console and write to file
            fprintf('%s', output_line);
            fprintf(output_file, '%s', output_line);

            % Store for plotting
            objective_values(end+1) = round(objective_value);
            instance_labels{end+1} = sprintf('gap%d-%d', g, p);
        end

        fclose(fid);
    end

    fclose(output_file);

    % Plotting results
    figure;
    plot(objective_values, '-o', 'LineWidth', 2);
    title('Optimal Fitness Value for Each GAP Instance');
    xlabel('Instance');
    ylabel('Objective Value');
    xticks(1:length(instance_labels));
    xticklabels(instance_labels);
    xtickangle(45);
    grid on;
end

% ILP Solver Function
function x_matrix = solve_gap_ilp(m, n, c, r, b)
    num_vars = m * n;
    f = -reshape(c, [num_vars, 1]);  % Negative for maximization
    intcon = 1:num_vars;
    lb = zeros(num_vars, 1);
    ub = ones(num_vars, 1);

    % User assignment constraint
    Aeq = zeros(n, num_vars);
    beq = ones(n, 1);
    for j = 1:n
        for i = 1:m
            idx = (j - 1) * m + i;
            Aeq(j, idx) = 1;
        end
    end

    % Server capacity constraint
    A = zeros(m, num_vars);
    b_ub = b;
    for i = 1:m
        for j = 1:n
            idx = (j - 1) * m + i;
            A(i, idx) = r(i, j);
        end
    end

    % Solve ILP
    options = optimoptions('intlinprog', 'Display', 'off');
    [x, ~, exitflag] = intlinprog(f, intcon, A, b_ub, Aeq, beq, lb, ub, options);

    if exitflag <= 0
        warning('No feasible solution found.');
        x_matrix = zeros(m, n);
    else
        x_matrix = reshape(round(x), [m, n]);
    end
end
