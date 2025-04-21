%2024PGCSCS17
function solve_large_gap_binary_ga()
    ga_results = [];
    gap12_convergences = {}; % Cell array to store convergence of all GAP12 problems

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

            [x_matrix, convergence] = solve_gap_binary_ga(m, n, c, r, b);
            objective_value = sum(sum(c .* x_matrix));
            ga_results(end+1, :) = [g + p/10, objective_value];

            fprintf('c%d-%d  %d\n', m*100 + n, p, round(objective_value));

            if g == 12
                gap12_convergences{end+1} = convergence;
            end
        end

        fclose(fid);
    end

    % Plot all GAP12 convergence curves together
    if ~isempty(gap12_convergences)
        figure;
        hold on;
        for i = 1:length(gap12_convergences)
            plot(gap12_convergences{i}, 'LineWidth', 1.5);
        end
        title('Binary GA Convergence for GAP12 Problem Instances');
        xlabel('Generation');
        ylabel('Best Fitness');
        legend(arrayfun(@(i) sprintf('GAP12-%d', i), 1:length(gap12_convergences), 'UniformOutput', false));
        grid on;
        hold off;
    end

    save('binary_ga.mat', 'ga_results');
end

function [x_matrix, convergence] = solve_gap_binary_ga(m, n, c, r, b)
    pop_size = 100;
    max_gen = 300;
    crossover_rate = 0.8;
    mutation_rate = 0.01;

    num_genes = m * n;

    % Initial binary population
    population = randi([0 1], pop_size, num_genes);
    for i = 1:pop_size
        population(i, :) = enforce_feasibility(population(i, :), m, n);
    end

    fitness = arrayfun(@(i) fitnessFcn(population(i, :)), 1:pop_size);
    convergence = zeros(1, max_gen);

    for gen = 1:max_gen
        parents = tournamentSelection(population, fitness);
        offspring = singlePointCrossover(parents, crossover_rate);
        mutated_offspring = bitFlipMutation(offspring, mutation_rate);

        for i = 1:size(mutated_offspring, 1)
            mutated_offspring(i, :) = enforce_feasibility(mutated_offspring(i, :), m, n);
        end

        new_fitness = arrayfun(@(i) fitnessFcn(mutated_offspring(i, :)), 1:size(mutated_offspring, 1));

        combined_population = [population; mutated_offspring];
        combined_fitness = [fitness, new_fitness];

        [~, sorted_idx] = sort(combined_fitness, 'descend');
        population = combined_population(sorted_idx(1:pop_size), :);
        fitness = combined_fitness(sorted_idx(1:pop_size));
        convergence(gen) = max(fitness);
    end

    [~, best_idx] = max(fitness);
    x_matrix = reshape(population(best_idx, :), [m, n]);

    function fval = fitnessFcn(x)
        x_mat = reshape(x, [m, n]);
        cost = sum(sum(c .* x_mat));
        capacity_violation = sum(max(sum(x_mat .* r, 2) - b, 0));
        assignment_violation = sum(abs(sum(x_mat, 1) - 1));
        penalty = 1e6 * (capacity_violation + assignment_violation);
        fval = cost - penalty;
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

function offspring = singlePointCrossover(parents, crossover_rate)
    [pop_size, num_genes] = size(parents);
    offspring = parents;
    for i = 1:2:pop_size-1
        if rand < crossover_rate
            point = randi([1 num_genes-1]);
            offspring(i, :) = [parents(i, 1:point), parents(i+1, point+1:end)];
            offspring(i+1, :) = [parents(i+1, 1:point), parents(i, point+1:end)];
        end
    end
end

function mutated = bitFlipMutation(offspring, mutation_rate)
    mutated = offspring;
    [rows, cols] = size(offspring);
    for i = 1:rows
        for j = 1:cols
            if rand < mutation_rate
                mutated(i, j) = 1 - mutated(i, j);
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