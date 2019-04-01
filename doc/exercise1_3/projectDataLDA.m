function [Z ] = projectDataLDA(X, w)

% You need to return the following variables correctly.
Z = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================

Z=w'*X'; 
%F  =[Z;Z].*((w/(w'*w))*ones(1,length(Z))); 

% =============================================================

end
