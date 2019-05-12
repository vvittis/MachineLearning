function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.
[nSamples, nFeat]  =size(X);
superClass = X;

% ============================================================
for j = 1:nFeat
    meanOfFeature             = mean(superClass(:,j));
    stdOfFeature              = std(superClass(:,j));
    normalizedSuperClass(:,j) = (superClass(:,j) - meanOfFeature)/stdOfFeature;
end

mu     = meanOfFeature;% mean of each column (feature)
sigma  = stdOfFeature;% standart deviation of each column
X_norm = normalizedSuperClass;% normalize each column independently

end
