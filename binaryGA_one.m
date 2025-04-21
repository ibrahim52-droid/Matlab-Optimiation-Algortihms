%2024PGCSCS17
clc; clear;

%% Parameters
nVar = 4;               % Number of variables
varMin = -10;           % Lower bound
varMax = 10;            % Upper bound
nBits = 16;             % Bits per variable
chromosomeLength = nBits * nVar;

popSize = 50;           % Population size
maxGen = 100;           % Maximum generations
pc = 0.7;               % Crossover probability
pm = 1/chromosomeLength; % Mutation probability

%% Initial Population
pop = randi([0 1], popSize, chromosomeLength);
bestFitness = zeros(maxGen, 1);

%% Main GA Loop
for gen = 1:maxGen
    % Decode binary to real values
    decoded = zeros(popSize, nVar);
    for i = 1:nVar
        startIdx = (i - 1) * nBits + 1;
        endIdx = i * nBits;
        binSegment = pop(:, startIdx:endIdx);
        decimalValue = bi2de(binSegment, 'left-msb');
        decoded(:, i) = varMin + (decimalValue / (2^nBits - 1)) * (varMax - varMin);
    end

    % Fitness Evaluation (Sphere Function)
    fitness = sum(decoded.^2, 2); % Minimize

    % Store Best Fitness
    [minFitness, minIndex] = min(fitness);
    bestFitness(gen) = minFitness;
    bestSolution = decoded(minIndex, :);

    % Selection (Tournament)
    newPop = zeros(size(pop));
    for i = 1:2:popSize
        p1 = TournamentSelect(fitness);
        p2 = TournamentSelect(fitness);
        
        % Crossover
        if rand < pc
            [child1, child2] = SinglePointCrossover(pop(p1, :), pop(p2, :));
        else
            child1 = pop(p1, :);
            child2 = pop(p2, :);
        end
        
        % Mutation
        child1 = Mutate(child1, pm);
        child2 = Mutate(child2, pm);
        
        newPop(i, :) = child1;
        newPop(i+1, :) = child2;
    end
    
    pop = newPop;
end

%% Results
disp('Best Solution Found:');
disp(bestSolution);
disp(['Minimum Value of Sphere Function: ', num2str(min(bestFitness))]);

% Plot convergence
figure;
plot(bestFitness, 'LineWidth', 2);
xlabel('Generation');
ylabel('Best Fitness');
title('Convergence of Binary-Coded GA on Sphere Function');
grid on;

%% Supporting Functions
function idx = TournamentSelect(fitness)
    k = 3; % Tournament size
    candidates = randperm(length(fitness), k);
    [~, best] = min(fitness(candidates));
    idx = candidates(best);
end

function [c1, c2] = SinglePointCrossover(p1, p2)
    point = randi([1 length(p1)-1]);
    c1 = [p1(1:point) p2(point+1:end)];
    c2 = [p2(1:point) p1(point+1:end)];
end

function mutated = Mutate(chromosome, pm)
    mutationMask = rand(size(chromosome)) < pm;
    mutated = xor(chromosome, mutationMask);
end
