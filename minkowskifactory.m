function M = minkowskifactory(p, q, m)
% Returns a manifold struct to optimize over n-by-m real matrices with 
% respect to the semi-Riemannian structure.
%
% function M = minkowskifactory(p, q)
% function M = minkowskifactory(p, q, m)
% function M = minkowskifactory(p, q, m)
% p:    Dimension of the Negative Definite Subspace
% q:    Dimension of the Poisitive Definte Subspace
%
% Returns M, a structure describing the Minkowski space of real matrices,
% equipped with the Minkowski norm and associated trace inner
% product with a signature matrix, as a manifold for Manopt.
%
% m and n in general can be vectors to handle multidimensional arrays.
% If either of m or n is a vector, they are concatenated as [m, n].
%
% Using this simple linear manifold, Manopt can be used to solve standard
% unconstrained optimization problems, for example in replacement of
% Matlab's fminunc, but with respect to a semi-Riemannian structure.
%
% See also: N/A

% Modified from euclideanfactory.m in Manopt: www.manopt.org.
% Modified by: Tingran Gao, Nov. 12, 2018.
%
%
    % The size can be defined using both m and n, or simply with m.
    % If m is a scalar, then n is implicitly 1.
    % This mimics the use of built-in Matlab functions such as zeros(...).
    if ~exist('m', 'var') || isempty(m)
        m = 1;
    end
    
    n = p + q;
    dimensions_vec = [n, m];
    signature_vec = [p, q];
    Ipq = eye(n);
    Ipq(1:p,1:p) = -eye(p);
    
    M.sigMat = Ipq;
    
    M.size = @() dimensions_vec;
    
    M.signature = signature_vec;
    
    M.name = @() sprintf('Minkowski space R^(%s,%s)', num2str(signature_vec));
    
    M.dim = @() prod(dimensions_vec);
    
    M.inner = @(x, d1, d2) d1.'*Ipq*d2;
    
    M.norm = @(x, d) M.inner(x, d, d);
    
%     M.dist = @(x, y) norm(x(:) - y(:), 'fro');
    
    % M.typicaldist = @() sqrt(prod(dimensions_vec));
    
    M.proj = @(x, d) d;
    
    M.egrad2rgrad = @(x, g) g;
    
    M.egrad2srgrad = @(x, g) Ipq*g;
    
    M.ehess2rhess = @(x, eg, eh, d) eh;
    
    M.tangent = M.proj;
    
    M.getDescDir = @getDescDir;
    function [descDir, descDirNorm] = getDescDir(x, srgrad, deterministic)
        if deterministic
            descDir = Ipq*srgrad;
            descDirNorm = sqrt(srgrad'*Ipq*descDir);
        else
            orthobasis = sr_tangentorthobasis(M, x);
            gradcoeff = M.inner(x, orthobasis, srgrad);
            descDir = orthobasis*gradcoeff;
            descDirNorm = norm(gradcoeff);
        end
    end
    
    M.exp = @exp;
    function y = exp(x, d, t)
        if nargin == 3
            y = x + t*d;
        else
            y = x + d;
        end
    end
    
    M.retr = M.exp;
	
	M.log = @(x, y) y-x;

    M.hash = @(x) ['z' hashmd5(x(:))];
    
    M.rand = @() randn(dimensions_vec);
    
    M.randvec = @randvec;
    function u = randvec(x) %#ok<INUSD>
        u = randn(dimensions_vec);
        u = u / norm(u(:), 'fro');
    end
    
    M.lincomb = @matrixlincomb;
    
    M.zerovec = @(x) zeros(dimensions_vec);
    
    M.transp = @(x1, x2, d) d;
    M.isotransp = M.transp; % the transport is isometric
    
    M.pairmean = @(x1, x2) .5*(x1+x2);
    
    M.vec = @(x, u_mat) u_mat(:);
    M.mat = @(x, u_vec) reshape(u_vec, dimensions_vec);
    M.vecmatareisometries = @() true;

end
