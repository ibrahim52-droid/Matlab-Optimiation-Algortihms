function solve_large_gap_real_ga_new()
    ga_results = [];
    gap12_convergences_real = {};

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

            [x_matrix, convergence] = solve_gap_real_ga(m, n, c, r, b);
            objective_value = sum(sum(c .* x_matrix));
            ga_results(end+1, :) = [g + p/10, objective_value];

            fprintf('c%d-%d  %d\n', m*100 + n, p, round(objective_value));

            if g == 12
                gap12_convergences_real{end+1} = convergence;
            end
        end

        fclose(fid);
    end

    % Plot convergence for GAP12
    if ~isempty(gap12_convergences_real)
        figure;
        hold on;
        for i = 1:length(gap12_convergences_real)
            plot(gap12_convergences_real{i}, 'LineWidth', 1.5);
        end
        title('GA Convergence for GAP12 Problem Instances');
        xlabel('Generation');
        ylabel('Best Fitness');
        legend(arrayfun(@(i) sprintf('GAP12-%d', i), 1:length(gap12_convergences_real), 'UniformOutput', false));
        grid on;
        hold off;
    end

    save('real_ga.mat', 'ga_results', 'gap12_convergences_real');

    %% === Compare Binary and Real GA ===
    if exist('binary_ga.mat', 'file') ~= 2
        error('binary_ga.mat not found. Please run solve_large_gap_binary_ga.m first.');
    end
    load('binary_ga.mat', 'gap12_convergences_binary');

    num_instances = length(gap12_convergences_real);

    figure;
    for i = 1:num_instances
        subplot(ceil(num_instances/3), 3, i);

        real_curve = gap12_convergences_real{i};
        binary_curve = gap12_convergences_binary{i};

        plot(binary_curve, 'r-', 'LineWidth', 1.5); hold on;
        plot(real_curve, 'g--', 'LineWidth', 1.5);

        title(sprintf('GAP12-%d', i));
        xlabel('Generation');
        ylabel('Best Fitness');
        legend({'Binary GA', 'Real GA'}, 'Location', 'best');
        grid on;
    end
    sgtitle('Convergence Comparison: Binary vs Real GA on GAP12');
end

function [x_matrix, convergence] = solve_gap_real_ga(m, n, c, r, b)
    pop_size = 100;
    max_gen = 300;
    crossover_rate = 0.8;
    mutation_rate = 0.1;
    num_genes = m * n;

    population = rand(pop_size, num_genes);
    for i = 1:pop_size
        population(i, :) = enforce_feasibility(population(i, :), m, n);
    end

    fitness = zeros(pop_size, 1);
    obj_values = zeros(pop_size, 1);

    for i = 1:pop_size
        [fitness(i), obj_values(i)] = fitnessFcn(population(i, :));
    end

    convergence = zeros(1, max_gen);

    for gen = 1:max_gen
        parents = tournamentSelection(population, fitness);
        offspring = arithmeticCrossover(parents, crossover_rate);
        mutated_offspring = gaussianMutation(offspring, mutation_rate);

        for i = 1:size(mutated_offspring, 1)
            mutated_offspring(i, :) = enforce_feasibility(mutated_offspring(i, :), m, n);
        end

        new_fitness = zeros(size(mutated_offspring, 1), 1);
        new_obj_values = zeros(size(mutated_offspring, 1), 1);

        for i = 1:size(mutated_offspring, 1)
            [new_fitness(i), new_obj_values(i)] = fitnessFcn(mutated_offspring(i, :));
        end

        combined_population = [population; mutated_offspring];
        combined_fitness = [fitness; new_fitness];
        combined_obj_values = [obj_values; new_obj_values];

        [~, sorted_idx] = sort(combined_fitness, 'descend');
        population = combined_population(sorted_idx(1:pop_size), :);
        fitness = combined_fitness(sorted_idx(1:pop_size));
        obj_values = combined_obj_values(sorted_idx(1:pop_size));

        % Track best feasible solution
        feasible_mask = (obj_values < 1e6);  % Filter out infeasible solutions (high penalty)
        if any(feasible_mask)
            convergence(gen) = min(obj_values(feasible_mask));  % Track the best feasible solution
        else
            convergence(gen) = NaN;  % No feasible solutions in this generation
        end
    end

    [~, best_idx] = max(fitness);
    x_matrix = reshape(population(best_idx, :) > 0.5, [m, n]);

    function [fval, obj] = fitnessFcn(x)
        x_mat = reshape(x > 0.5, [m, n]);
        total_cost = sum(sum(c .* x_mat));
        cap_violation = sum(max(sum(x_mat .* r, 2) - b, 0));
        assign_violation = sum(abs(sum(x_mat, 1) - 1));

        penalty = 1e4 * cap_violation + 1e4 * assign_violation;
        fval = 1e6 - (total_cost + penalty);  % Higher is better
        obj = total_cost + penalty;          % Save real cost to track
    end
end

function selected = tournamentSelection(population, fitness)
    pop_size = size(population, 1);
    selected = zeros(size(population));
    for i = 1:pop_size
        idx1 = randi(pop_size);
        idx2 = randi(pop_size);
        if fitness(idx1) > fitness(idx2)
            selected(i, :) = population(idx1, :);
        else
            selected(i, :) = population(idx2, :);
        end
    end
end

function offspring = arithmeticCrossover(parents, crossover_rate)
    [pop_size, num_genes] = size(parents);
    offspring = parents;
    for i = 1:2:pop_size-1
        if rand < crossover_rate
            alpha = rand;
            offspring(i, :) = alpha * parents(i, :) + (1 - alpha) * parents(i+1, :);
            offspring(i+1, :) = alpha * parents(i+1, :) + (1 - alpha) * parents(i, :);
        end
    end
end

function mutated = gaussianMutation(offspring, mutation_rate)
    [rows, cols] = size(offspring);
    mutated = offspring;
    for i = 1:rows
        for j = 1:cols
            if rand < mutation_rate
                mutated(i, j) = mutated(i, j) + 0.1 * randn;
                mutated(i, j) = min(max(mutated(i, j), 0), 1);
            end
        end
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
