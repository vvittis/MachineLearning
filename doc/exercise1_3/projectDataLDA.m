function [Z] = projectDataLDA(X, w)

% You need to return the following variables correctly.
Z = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================

Z=w'*X'; 
% Z=[t1;t1].*((w/(w'*w))*ones(1,length(t1))); 

% =============================================================

end
