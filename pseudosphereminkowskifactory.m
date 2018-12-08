function M = pseudosphereminkowskifactory(p, q)
% Returns a manifold struct to optimize over pseudospheres in an Minkowski
% space of signature (p,q).
%
% function M = pseudosphereminkowskifactory(p, q)
%
% last modified: Dec 6, 2018
% Tingran Gao (tingrangao@galton.uchicago.edu)
%
    
    array_type = 'double';
    
    n = p + q;
    signature_vec = [p, q];
    Ipq = eye(n);
    Ipq(1:p,1:p) = -eye(p);
    
    M.sigMat = Ipq; %% this is the signature matrix of the ambient Minkowski space
    
    M.signature = signature_vec;
    
    M.name = @() sprintf('Psuedo-sphere S^(%d,%d)', p, q);
    
    M.dim = @() n-1;
    
    M.inner = @(x, d1, d2) d1.'*Ipq*d2;
    
    M.norm = @(x, d) M.inner(x, d, d);
    
    M.proj = @pseudosphere_proj;
    function rslt = pseudosphere_proj(x, d)
        rslt = d - (((Ipq*x(:))'*d(:))/(x'*Ipq*x))*x;  
%         rslt = d - (((Ipq*x(:))'*d(:))/(norm(Ipq*x)))*x;
    end
    
%     M.proj = @sphere_semiriem_proj;
%     function rslt = sphere_semiriem_proj(x, d)
%         rslt = d - ((x(:)'*d(:))/(x'*Ipq*x))*Ipq*x;
%     end
    
    M.tangent = M.proj;
    
    M.getDescDir = @getDescDir;
    function [descDir, descDirNorm] = getDescDir(x, srgrad, deterministic)
        orthobasis = sr_tangentorthobasis(M, x);
        gradcoeff = M.inner(x, orthobasis, srgrad);
        descDir = orthobasis*gradcoeff;
        descDirNorm = norm(gradcoeff);
    end
    
    % For Riemannian submanifolds, converting a Euclidean gradient into a
    % Riemannian gradient amounts to an orthogonal projection; for
    % semi-Riemannian submanifolds, the conversion is similar but with
    % respect to a different metric
    M.egrad2srgrad = @(x, g) M.proj(x, Ipq * g);
        
%     M.ehess2rhess = @ehess2rhess;
%     function rhess = ehess2rhess(x, egrad, ehess, u)
%         rhess = M.proj(x, ehess - (x(:)'*egrad(:))*u);
%     end
    
    M.exp = @(x, d, t) exponential(x, d, t, Ipq);
    M.retr = @(x, d, t) exponential(x, d, t, Ipq);
%     M.retr = @exponential;
%     M.retr = @retraction;
%     M.invretr = @inverse_retraction;

%     M.log = @logarithm;
%     function v = logarithm(x1, x2)
%         v = M.proj(x1, x2 - x1);
%         di = M.dist(x1, x2);
%         % If the two points are "far apart", correct the norm.
%         if di > 1e-6
%             nv = norm(v, 'fro');
%             v = v * (di / nv);
%         end
%     end
    
    M.hash = @(x) ['z' hashmd5(x(:))];
    
%     M.rand = @() random(n, 1, array_type);
    
    M.randvec = @(x) randomvec(n, 1, x, Ipq);
    
    M.zerovec = @(x) zeros(n, 1, array_type);
    
    M.lincomb = @matrixlincomb;
    
    M.transp = @(x1, x2, d) M.proj(x2, d);
        
    M.pairmean = @pairmean;
    function y = pairmean(x1, x2)
        y = x1+x2;
        y((p+1):end) = (y((p+1):end)/norm(y((p+1):end)))*sqrt(1+norm(y(1:p))^2);
%         y = y / norm(y, 'fro');
    end

    M.vec = @(x, u_mat) u_mat(:);
    M.mat = @(x, u_vec) reshape(u_vec, [n, m]);
    M.vecmatareisometries = @() true;
    
end

% Exponential on the pseudo-sphere
function y = exponential(x, d, t, Ipq)

    if nargin == 2
        % t = 1
        td = d;
    else
        td = t*d;
    end
    
    nrm_td = sqrt(abs(td'*Ipq*td));
%     nrm_td = norm(td, 'fro');
    
    if nrm_td > 0
        if td(:)'*Ipq*td(:) > 0
            y = x*cos(nrm_td) + td*(sin(nrm_td)/nrm_td);
        elseif td(:)'*Ipq*td(:) < 0
            y = x*cosh(nrm_td) + td*(sinh(nrm_td)/nrm_td);
        else
            y = x + td;
        end
    else
        y = x;
    end
    
    if isnan(y(1)) || isinf(y(1))
        y = exponential(x, d, t/2, Ipq);
    end

end

% function y = exponential(x, d, t)
% 
%     if nargin == 2
%         % t = 1
%         td = d;
%     else
%         td = t*d;
%     end
%     
%     nrm_td = norm(td, 'fro');
%     
%     % Former versions of Manopt avoided the computation of sin(a)/a for
%     % small a, but further investigations suggest this computation is
%     % well-behaved numerically.
%     if nrm_td > 0
%         y = x*cos(nrm_td) + td*(sin(nrm_td)/nrm_td);
%     else
%         y = x;
%     end
% 
% end

% % Retraction on the sphere
% function y = retraction(x, d, t)
% 
%     if nargin == 2
%         % t = 1;
%         td = d;
%     else
%         td = t*d;
%     end
%     
%     y = x + td;
%     y = y / norm(y, 'fro');
% 
% end

% Given x and y two points on the manifold, if there exists a tangent
% vector d at x such that Retr_x(d) = y, this function returns d.
% function d = inverse_retraction(x, y)
% 
%     % Since
%     %   x + d = y*||x + d||
%     % and x'd = 0, multiply the above by x' on the left:
%     %   1 + 0 = x'y * ||x + d||
%     % Then solve for d:
%     
%     d = y/(x(:)'*y(:)) - x;
% 
% end

% % Uniform random sampling on the sphere.
% function x = random(n, m, array_type)
% 
%     x = randn(n, m, array_type);
%     x = x / norm(x, 'fro');
% 
% end

% Random normalized tangent vector at x.
function d = randomvec(n, m, x, Ipq)

    d = randn(n, m);
    d = d - (((Ipq*x(:))'*d(:))/(x'*Ipq*x))*x;
%     d = d - x*(x(:)'*d(:));
    d = d / norm(d, 'fro');

end
