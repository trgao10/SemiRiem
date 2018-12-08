function Y = sr_orthogonalize(M, x, X)

m = size(X,2);
Y = zeros(size(X));

Ipq = M.sigMat;
norm1sq = X(:,1)' * Ipq * X(:,1);
norm1 = sqrt(abs(norm1sq));
Y(:,1) = X(:,1) / norm1;

for c=2:m
    signDiag = diag(Y(:,1:(c-1))'*Ipq*Y(:,1:(c-1)));
    Y(:,c) = X(:,c) - Y(:,1:(c-1))*((Y(:,1:(c-1))'*Ipq*X(:,c)).*signDiag);
    normcsq = Y(:,c)' * Ipq * Y(:,c);
    normc = sqrt(abs(normcsq));
    Y(:,c) = Y(:,c) / normc;
end


end
