%2024PGCSCS17
clc;
clear;

% Sphere Function
f = @(x) sum(x.^2);

% GA Parameters
nVar = 4;                  % Number of decision variables
VarSize = [1 nVar];        % Size of chromosome
VarMin = -10;              % Lower bound
VarMax = 10;               % Upper bound

MaxIt = 100;               % Maximum number of iterations (generations)
nPop = 50;                 % Population size
pc = 0.7;                  % Crossover percentage
nc = 2*round(pc*nPop/2);   % Number of children (must be even)
pm = 0.3;                  % Mutation percentage
nm = round(pm*nPop);       % Number of mutants
mu = 0.02;                 % Mutation rate

% Initialize Population
empty_individual.Position = [];
empty_individual.Cost = [];

pop = repmat(empty_individual, nPop, 1);
for i = 1:nPop
    pop(i).Position = unifrnd(VarMin, VarMax, VarSize);
    pop(i).Cost = f(pop(i).Position);
end

% Sort Population
[~, SortOrder] = sort([pop.Cost]);
pop = pop(SortOrder);

% Best Solution Ever Found
BestSol = pop(1);

% Array to Hold Best Cost Value
BestCost = zeros(MaxIt, 1);

% Main Loop
for it = 1:MaxIt
    
    % Crossover
    popc = repmat(empty_individual, nc/2, 2);
    for k = 1:nc/2
        i1 = randi([1 nPop]);
        i2 = randi([1 nPop]);
        p1 = pop(i1);
        p2 = pop(i2);
        
        [popc(k,1).Position, popc(k,2).Position] = crossover(p1.Position, p2.Position);
        popc(k,1).Cost = f(popc(k,1).Position);
        popc(k,2).Cost = f(popc(k,2).Position);
    end
    popc = popc(:);
    
    % Mutation
    popm = repmat(empty_individual, nm, 1);
    for k = 1:nm
        i = randi([1 nPop]);
        p = pop(i);
        popm(k).Position = mutate(p.Position, mu, VarMin, VarMax);
        popm(k).Cost = f(popm(k).Position);
    end
    
    % Merge Populations
    pop = [pop; popc; popm]; %#ok<AGROW>
    
    % Sort Population
    [~, SortOrder] = sort([pop.Cost]);
    pop = pop(SortOrder);
    
    % Truncate to Original Population Size
    pop = pop(1:nPop);
    
    % Update Best Solution
    BestSol = pop(1);
    
    % Store Best Cost
    BestCost(it) = BestSol.Cost;
    
    % Display Iteration Information
    fprintf('Iteration %d: Best Cost = %.6f\n', it, BestCost(it));
end

% Plot Results
figure;
plot(BestCost, 'LineWidth', 2);
xlabel('Iteration');
ylabel('Best Cost');
grid on;

% Display Optimal Solution
disp('Optimal x:');
disp(BestSol.Position);
disp('Minimum value:');
disp(BestSol.Cost);

% --- Crossover Function ---
function [y1, y2] = crossover(x1, x2)
    alpha = rand(size(x1));
    y1 = alpha.*x1 + (1 - alpha).*x2;
    y2 = alpha.*x2 + (1 - alpha).*x1;
end

% --- Mutation Function ---
function y = mutate(x, mu, VarMin, VarMax)
    nVar = numel(x);
    nMut = ceil(mu * nVar);
    j = randperm(nVar, nMut);
    y = x;
    y(j) = x(j) + randn(size(j)) .* (VarMax - VarMin)/10;
    y = max(min(y, VarMax), VarMin);
end
