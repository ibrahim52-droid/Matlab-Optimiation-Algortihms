%2024PGCSCS17
function pso_sphere()
    % PSO parameters
    num_particles = 30;       % Number of particles
    num_dimensions = 4;      % Number of variables (4 for Sphere function)
    max_iter = 1000;         % Maximum number of iterations
    w = 0.7;                 % Inertia weight
    c1 = 1.5;                % Cognitive coefficient
    c2 = 1.5;                % Social coefficient
    lower_bound = -10;       % Lower bound for the search space
    upper_bound = 10;        % Upper bound for the search space
    
    % Initialize particle positions and velocities
    particles = lower_bound + (upper_bound - lower_bound) * rand(num_particles, num_dimensions);
    velocities = rand(num_particles, num_dimensions);
    
    % Initialize personal best positions and global best position
    pbest = particles;
    pbest_score = inf * ones(num_particles, 1);  % Initialize with large values
    [gbest_score, gbest_index] = min(pbest_score);
    gbest = pbest(gbest_index, :);

    % Array to store the convergence values
    convergence = zeros(max_iter, 1);

    % PSO main loop
    for iter = 1:max_iter
        % Evaluate fitness of each particle (Sphere function)
        for i = 1:num_particles
            fitness = sphere_function(particles(i, :));
            
            % Update personal best
            if fitness < pbest_score(i)
                pbest(i, :) = particles(i, :);
                pbest_score(i) = fitness;
            end
        end
        
        % Update global best
        [best_score, best_index] = min(pbest_score);
        if best_score < gbest_score
            gbest = pbest(best_index, :);
            gbest_score = best_score;
        end
        
        % Store the current global best score for convergence plot
        convergence(iter) = gbest_score;
        
        % Update velocities and positions
        for i = 1:num_particles
            r1 = rand(1, num_dimensions);
            r2 = rand(1, num_dimensions);
            velocities(i, :) = w * velocities(i, :) + c1 * r1 .* (pbest(i, :) - particles(i, :)) + c2 * r2 .* (gbest - particles(i, :));
            particles(i, :) = particles(i, :) + velocities(i, :);
            
            % Ensure particles stay within bounds
            particles(i, :) = max(min(particles(i, :), upper_bound), lower_bound);
        end
        
        % Display progress
        fprintf('Iteration %d, Global Best Fitness: %f\n', iter, gbest_score);
        
        % Check for convergence (optional termination condition)
        if gbest_score <= 1e-6  % You can adjust this tolerance
            fprintf('Convergence reached with score: %f\n', gbest_score);
            break;
        end
    end

    % Display final result
    fprintf('Final global best score: %f\n', gbest_score);
    fprintf('Global best solution: [%s]\n', num2str(gbest));
    
    % Plot the convergence graph
    figure;
    plot(convergence(1:iter)); % Plot up to the current iteration
    title('Convergence of PSO');
    xlabel('Iteration');
    ylabel('Global Best Fitness Value');
    grid on;
end

% Sphere function definition
function f = sphere_function(x)
    f = sum(x.^2);  % Sphere function: f(x) = sum(x_i^2)
end
