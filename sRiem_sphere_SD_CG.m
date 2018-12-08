close all
clear all
clc

% Generate a random quadratic problem
rng(20181126);

n = 10;
A = randn(n);
A = .5*(A+A.');
x0 = rand(n,1);
x0 = x0 / norm(x0);

%% Steepest Descent

disp('+++++++++++++++++++++ Steepest Descent +++++++++++++++++++++++++');
options.statsfun = @ptTrace;
%%%% solve it once with trusted regions as "ground truth"
problem.M = spherefactory(n);
problem.cost = @(x) -x'*A*x;
problem.egrad = @(x) -2*A*x;

[x_truth, xcost_truth, info_truth, options_truth] = trustregions(problem);
disp(['ground truth: [' num2str(x_truth') ']']);

%%%% solve with steepest descent: p=0 (Riemannian optimization)
options.verbosity = 0;
options.deterministic = true;
options.statsfun = @ptTrace;
options.linesearch = @linesearch_hint;
options.beta_type = 'P-R';
% options.maxiter = 10;
options.ls_suff_decr = 0.1;
[x, xcost, info, options] = steepestdescent(problem, x0, options);
disp(['Riemannian optimal solution: [' num2str(x') ']']);

figure('Position', [0,0,800,800]);
set(gca,'FontSize',20);

err00 = sqrt(sum(([info(:).x]-repmat(x_truth, 1, size([info(:).x],2))).^2));
err01 = sqrt(sum(([info(:).x]+repmat(x_truth, 1, size([info(:).x],2))).^2));
err0 = min(err00, err01);
semilogy([info.iter], err0, 'd-');
grid on
grid minor
xlabel('Number of Iterations', 'Interpreter', 'Latex', 'FontSize', 25);
ylabel('Error', 'Interpreter', 'Latex', 'FontSize', 25);
title('Steepest Descent', 'Interpreter', 'Latex', 'FontSize', 30);
set(gca,'TickLabelInterpreter', 'tex');
hold on

%%% p=1, q=n-1
p = 1;
q = n-p;
problem2.M = sphereminkowskifactory(p,q);
problem2.cost = problem.cost;
problem2.egrad = problem.egrad;

options.linesearch = @sr_linesearch_hint;
[x2, xcost2, info2, options] = sr_steepestdescent(problem2, x0, options);
disp(['semi-Riemannian optimal solution: [' num2str(x2') ']']);

err10 = sqrt(sum(([info2(:).x]-repmat(x_truth, 1, size([info2(:).x],2))).^2));
err11 = sqrt(sum(([info2(:).x]+repmat(x_truth, 1, size([info2(:).x],2))).^2));
err1 = min(err10, err11);
semilogy([info2.iter], err1, '*-');

%%% p=2, q=n-2
p = 2;
q = n-p;
problem2.M = sphereminkowskifactory(p,q);
problem2.cost = problem.cost;
problem2.egrad = problem.egrad;

options.linesearch = @sr_linesearch_hint;
[x2, xcost2, info2, options] = sr_steepestdescent(problem2, x0, options);
disp(['semi-Riemannian optimal solution: [' num2str(x2') ']']);

err10 = sqrt(sum(([info2(:).x]-repmat(x_truth, 1, size([info2(:).x],2))).^2));
err11 = sqrt(sum(([info2(:).x]+repmat(x_truth, 1, size([info2(:).x],2))).^2));
err1 = min(err10, err11);
semilogy([info2.iter], err1, '+-');

%%% p=3, q=n-3
p = 3;
q = n-p;
problem2.M = sphereminkowskifactory(p,q);
problem2.cost = problem.cost;
problem2.egrad = problem.egrad;

options.linesearch = @sr_linesearch_hint;
[x2, xcost2, info2, options] = sr_steepestdescent(problem2, x0, options);
disp(['semi-Riemannian optimal solution: [' num2str(x2') ']']);

err10 = sqrt(sum(([info2(:).x]-repmat(x_truth, 1, size([info2(:).x],2))).^2));
err11 = sqrt(sum(([info2(:).x]+repmat(x_truth, 1, size([info2(:).x],2))).^2));
err1 = min(err10, err11);
semilogy([info2.iter], err1, 'x-');

%%% p=4, q=n-4
p = 4;
q = n-p;
problem2.M = sphereminkowskifactory(p,q);
problem2.cost = problem.cost;
problem2.egrad = problem.egrad;

options.linesearch = @sr_linesearch_hint;
[x2, xcost2, info2, options] = sr_steepestdescent(problem2, x0, options);
disp(['semi-Riemannian optimal solution: [' num2str(x2') ']']);

err10 = sqrt(sum(([info2(:).x]-repmat(x_truth, 1, size([info2(:).x],2))).^2));
err11 = sqrt(sum(([info2(:).x]+repmat(x_truth, 1, size([info2(:).x],2))).^2));
err1 = min(err10, err11);
semilogy([info2.iter], err1, 'o-');

legend({'$p=0, q=10$', '$p=1, q=9$', '$p=2, q=8$', '$p=3, q=7$', '$p=4, q=6$'}, 'Interpreter', 'Latex', 'FontSize', 22);


%%
disp('+++++++++++++++++++++ Conjugate Gradient +++++++++++++++++++++++++');

problem.M = spherefactory(n);
problem.cost = @(x) -x'*A*x;
problem.egrad = @(x) -2*A*x;

options.statsfun = @ptTrace;
options.linesearch = @linesearch_hint;
[x, xcost, info, options] = conjugategradient(problem, x0, options);
disp(['Riemannnian optimal solution: ' num2str(x') ']']);

figure('Position', [0,0,800,800]);
set(gca,'FontSize',20);

err00 = sqrt(sum(([info(:).x]-repmat(x_truth, 1, size([info(:).x],2))).^2));
err01 = sqrt(sum(([info(:).x]+repmat(x_truth, 1, size([info(:).x],2))).^2));
err0 = min(err00, err01);
semilogy([info.iter], err0, 'd-');
grid on
grid minor
xlabel('Number of Iterations', 'Interpreter', 'Latex', 'FontSize', 25);
ylabel('Error', 'Interpreter', 'Latex', 'FontSize', 25);
title('Conjugate Gradient', 'Interpreter', 'Latex', 'FontSize', 30);
hold on

%%% p=1, q=n-1
p = 1;
q = n-1;
problem2.M = sphereminkowskifactory(p,q);
problem2.cost = problem.cost;
problem2.egrad = problem.egrad;

options.linesearch = @sr_linesearch_hint;
[x2, xcost2, info2, options] = sr_conjugategradient(problem2, x0, options);
disp(['semi-Riemannian optimal solution: ' num2str(x2') ']']);

err10 = sqrt(sum(([info2(:).x]-repmat(x_truth, 1, size([info2(:).x],2))).^2));
err11 = sqrt(sum(([info2(:).x]+repmat(x_truth, 1, size([info2(:).x],2))).^2));
err1 = min(err10, err11);
semilogy([info2.iter], err1, '*-');

%%% p=2, q=n-2
p = 2;
q = n-p;
problem2.M = sphereminkowskifactory(p,q);
problem2.cost = problem.cost;
problem2.egrad = problem.egrad;

options.linesearch = @sr_linesearch_hint;
[x2, xcost2, info2, options] = sr_conjugategradient(problem2, x0, options);
disp(['semi-Riemannian optimal solution: [' num2str(x2') ']']);

err10 = sqrt(sum(([info2(:).x]-repmat(x_truth, 1, size([info2(:).x],2))).^2));
err11 = sqrt(sum(([info2(:).x]+repmat(x_truth, 1, size([info2(:).x],2))).^2));
err1 = min(err10, err11);
semilogy([info2.iter], err1, '+-');

%%% p=3, q=n-3
p = 3;
q = n-p;
problem2.M = sphereminkowskifactory(p,q);
problem2.cost = problem.cost;
problem2.egrad = problem.egrad;

options.linesearch = @sr_linesearch_hint;
[x2, xcost2, info2, options] = sr_conjugategradient(problem2, x0, options);
disp(['semi-Riemannian optimal solution: [' num2str(x2') ']']);

err10 = sqrt(sum(([info2(:).x]-repmat(x_truth, 1, size([info2(:).x],2))).^2));
err11 = sqrt(sum(([info2(:).x]+repmat(x_truth, 1, size([info2(:).x],2))).^2));
err1 = min(err10, err11);
semilogy([info2.iter], err1, 'x-');

%%% p=4, q=n-4
p = 4;
q = n-p;
problem2.M = sphereminkowskifactory(p,q);
problem2.cost = problem.cost;
problem2.egrad = problem.egrad;

options.linesearch = @sr_linesearch_hint;
[x2, xcost2, info2, options] = sr_conjugategradient(problem2, x0, options);
disp(['semi-Riemannian optimal solution: [' num2str(x2') ']']);

err10 = sqrt(sum(([info2(:).x]-repmat(x_truth, 1, size([info2(:).x],2))).^2));
err11 = sqrt(sum(([info2(:).x]+repmat(x_truth, 1, size([info2(:).x],2))).^2));
err1 = min(err10, err11);
semilogy([info2.iter], err1, 'o-');

legend({'$p=0, q=10$', '$p=1, q=9$', '$p=2, q=8$', '$p=3, q=7$', '$p=4, q=6$'}, 'Interpreter', 'Latex', 'FontSize', 22);

%% auxiliary function
function stats = ptTrace(problem, x, stats)
    stats.x = x;
end
