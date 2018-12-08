close all
clear all
clc

% Generate a random quadratic problem
m = 1000;
n = 2;

rng(20181115);
% 
% A = randn(2);
% A = .5*(A+A');
% [V,D] = eig(A);
% A = V'*abs(D)*V;
% A = (A+A.')/2;
% x0 = rand(2,1)*2-1;

%%% the following is used in the paper
% A = [0.3649,-0.1065;-0.1065,1.7427];
% x0 = [-0.7285,0.0230]';

A = [0.5094,-0.3059;-0.3059,0.9427];
x0 = [0.5349,-0.5224]';

x = linspace(-1,1,m);
y = linspace(-1,1,m);
[X,Y] = meshgrid(x,y);

Z = reshape(sum(([X(:),Y(:)]*A).*[X(:),Y(:)],2),m,m);

%% Steepest Descent

disp('+++++++++++++++++++++ Steepest Descent +++++++++++++++++++++++++');

problem.M = euclideanfactory(n,1);
problem.cost = @(x) x'*A*x/2;
problem.egrad = @(x) A*x;

p = 1;
q = 1;
problem2.M = minkowskifactory(p,q);
problem2.cost = problem.cost;
problem2.egrad = problem.egrad;

options.deterministic = false;
options.statsfun = @ptTrace;
options.linesearch = @linesearch_hint;
% options.maxiter = 2;
% options.ls_suff_decr = 0.5;
[x, xcost, info, options] = steepestdescent(problem, x0, options);

options.linesearch = @sr_linesearch_hint;
[x2, xcost2, info2, options] = sr_steepestdescent(problem2, x0, options);

% make plot
figure('Position', [0,0,800,800]);
% hold on
% plot([-1,1],[-1,1],'k');
% plot([-1,1],[1,-1],'k');
% subplot(1,2,1);
hold on

%%% Riemannian path
traj = [info(:).x];
line(traj(1,:),traj(2,:),'LineWidth',2,'Marker','o','Color','b','MarkerSize',10);

%%% Semi-Riemannian path
straj = [info2(:).x];
line(straj(1,:),straj(2,:),'LineWidth',2,'Marker','.','Color','r','MarkerSize',15);
% plot(straj(1,end),straj(2,end),'r+','Linewidth',2,'MarkerSize',20);

plot(x0(1),x0(2),'k*','Linewidth',1.5,'MarkerSize',10);
plot(traj(1,end),traj(2,end),'kx','Linewidth',2,'MarkerSize',10);

legend({'Riemannian', 'Semi-Riemannian', 'Initialization', 'Global Optimum'}, 'Interpreter', 'Latex', 'FontSize', 15);

contour(X,Y,Z,30,'HandleVisibility','off');
axis equal
grid on
grid minor

title('Steepest Descent', 'Interpreter', 'Latex', 'FontSize', 30);

disp('+++++++++++++++++++++ Conjugate Gradient +++++++++++++++++++++++++');

problem.M = euclideanfactory(n,1);
problem.cost = @(x) x'*A*x/2;
problem.egrad = @(x) A*x;
problem.ehess = @(x,y) A*y;

p = 1;
q = 1;
problem2.M = minkowskifactory(p,q);
problem2.cost = problem.cost;
problem2.egrad = problem.egrad;

% options.statsfun = @ptTrace;
options.linesearch = @linesearch_hint;
[x, xcost, info, options] = conjugategradient(problem, x0, options);

options.linesearch = @sr_linesearch_hint;
[x2, xcost2, info2, options] = sr_conjugategradient(problem2, x0, options);

figure('Position', [800,0,800,800]);
% subplot(1,2,2);
hold on
set(gca,'TickLabelInterpreter', 'tex');

%%% Riemannian path
traj = [info(:).x];
line(traj(1,:),traj(2,:),'LineWidth',2,'Marker','o','Color','b','MarkerSize',10);

%%% Semi-Riemannian path
straj = [info2(:).x];
line(straj(1,:),straj(2,:),'LineWidth',2,'Marker','.','Color','r','MarkerSize',15);
% plot(straj(1,end),straj(2,end),'r+','Linewidth',2,'MarkerSize',20);

plot(x0(1),x0(2),'k*','Linewidth',1.5,'MarkerSize',10);
plot(traj(1,end),traj(2,end),'kx','Linewidth',2,'MarkerSize',10);

legend({'Riemannian', 'Semi-Riemannian', 'Initialization', 'Global Optimum'}, 'Interpreter', 'Latex', 'FontSize', 15);

contour(X,Y,Z,30,'HandleVisibility','off');
axis equal
grid on
grid minor

title('Conjugate Gradient', 'Interpreter', 'Latex', 'FontSize', 30);

function stats = ptTrace(problem, x, stats)
    stats.x = x;
end
