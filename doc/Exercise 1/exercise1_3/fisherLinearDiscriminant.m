function v = fisherLinearDiscriminant(X1, X2)

    m1 = size(X1, 1);
    m2 = size(X2, 1);
    
    mu1 = mean(X1)';  % mean value of X1
    mu2 = mean(X2)'; % mean value of X2
    
    S1=zeros(2);
    P1=sum(m1)/(m1+m2);
    S1=S1+P1*cov(X1);     % scatter matrix of X1
    
    S2=zeros(2);
    P2=sum(m2)/(m1+m2);
    S2=S2+P2*cov(X2);   % scatter matrix of X2
 
    Sw = S1 + S2;         % Within class scatter matrix

    v=inv(Sw)*(mu2 - mu1); % optimal direction for maximum class separation 
    
    [sorted_eigenvalues,ind] = sort(v,1,'descend'); %Sort them 

    v = sorted_eigenvalues/norm(sorted_eigenvalues);% return a vector of unit norm

