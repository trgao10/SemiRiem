close all
clear all
clc

% Generate a random quadratic problem
rng(20181208);

n = 15;
p = 3;
q = n-p;
Ipq = eye(n);
Ipq(1:p,1:p) = -eye(p);

%%% generate ground truth point
x_truth = randn(n,1);
x_truth((p+1):end) = (x_truth((p+1):end)/norm(x_truth((p+1):end)))*sqrt(1+norm(x_truth(1:p))^2);
disp(['x_truth = [' num2str(x_truth') ']']);

%%% generate a point outside of the pseudo-sphere to be projected
pt = x_truth+0.2*Ipq*x_truth;

%%% generate initial value
x0 = randn(n,1);
x0((p+1):end) = (x0((p+1):end)/norm(x0((p+1):end)))*sqrt(1+norm(x0(1:p))^2);
if sign(x0(end)) ~= sign(x_truth(end))
    x0((p+1):end) = -x0((p+1):end);
end
disp(['x0 = [' num2str(x0') ']']);

%%% run standard constrained optimization with MATLAB
H = {Ipq};
options_matlab = optimoptions(@fmincon,'Algorithm','interior-point',...
    'SpecifyObjectiveGradient',true,'SpecifyConstraintGradient',true,...
    'HessianFcn',@(x,lambda)quadhess(x,lambda,H));
fun = @(x) quadobj(x,pt);
nonlconstr = @(x)quadconstr(x,H);
[x_min,fval,eflag,output,lambda] = fmincon(fun,x0,...
    [],[],[],[],[],[],nonlconstr,options_matlab);
disp(['fmincon sol = [' num2str(x_min') ']']);

%% Steepest Descent

disp('+++++++++++++++++++++ Steepest Descent +++++++++++++++++++++++++');
options.statsfun = @ptTrace;

%%%% solve with steepest descent: p=0 (Riemannian optimization)
options = [];
options.verbosity = 0;
options.deterministic = true;
options.statsfun = @ptTrace;
options.linesearch = @sr_linesearch_hint;
options.ls_suff_decr = 0.1;

problem.M = pseudosphereminkowskifactory(p,q);
problem.cost = @(x) norm(x-pt)^2;
problem.egrad = @(x) 2*(x-pt);

[x_sr, xcost, info, options] = sr_steepestdescent(problem, x0, options);
% disp(['Semi-Riemannian optimal solution: [' num2str(x_sr') ']']);

%%
disp('+++++++++++++++++++++ Conjugate Gradient +++++++++++++++++++++++++');
options.verbosity = 0;
% options.maxiter = 10;
[x2, xcost2, info2, options] = sr_conjugategradient(problem, x0, options);

disp(['true distance = ' num2str(norm(pt-x_truth)^2)]);
disp(['fmincon cost = ' num2str(fval)]);
disp(['Semi-Riemannian SD cost: ' num2str(xcost)]);
disp(['Semi-Riemannian CG cost: ' num2str(xcost2)]);

%%
figure('Position', [0,0,1600,800]);
set(gca,'FontSize',20);

err1 = sqrt(sum(([info(:).x]-repmat(x_truth, 1, size([info(:).x],2))).^2));
err2 = sqrt(sum(([info2(:).x]-repmat(x_truth, 1, size([info2(:).x],2))).^2));
semilogy([info.iter], err1, 'bd-', 'LineWidth', 2, 'MarkerSize', 2);
hold on
semilogy([info2.iter], err2, 'ro-', 'LineWidth', 2, 'MarkerSize', 2);
grid on
grid minor
xlim([0,200]);
xlabel('Number of Iterations $k$', 'Interpreter', 'Latex', 'FontSize', 25);
ylabel('$\left\|x_k-x_{\textrm{true}}\right\|^2$', 'Interpreter', 'Latex', 'FontSize', 25);
% title('Steepest Descent', 'Interpreter', 'Latex', 'FontSize', 30);
set(gca,'TickLabelInterpreter', 'tex');
legend({'Semi-Riemannian Steepest Descent', 'Semi-Riemannian Conjugate Gradient'}, 'Interpreter', 'Latex', 'FontSize', 25);

%% auxiliary functions
function stats = ptTrace(problem, x, stats)
    stats.x = x;
end

function [y,grady] = quadobj(x, pt)
    y = norm(x-pt)^2;
    if nargout > 1
        grady = 2*(x-pt);
    end
end

function [y,yeq,grady,gradyeq] = quadconstr(x, H)
    jj = length(H); % jj is the number of inequality constraints
    yeq = zeros(1,jj);
    for i = 1:jj
        yeq(i) = x'*H{i}*x-1;
    end
    y = [];
    
    if nargout > 2
        gradyeq = zeros(length(x),jj);
        for i = 1:jj
            gradyeq(:,i) = 2*H{i}*x;
        end
    end
    grady = [];
end

function hess = quadhess(x,lambda,H)
    hess = eye(length(x));
    jj = length(H); % jj is the number of inequality constraints
    for i = 1:jj
        hess = hess + lambda.eqnonlin(i)*H{i};
    end
end

