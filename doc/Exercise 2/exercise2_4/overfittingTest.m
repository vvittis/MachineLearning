% overfittingTest.m

% -------------PART A-------------------------------------------------------
numFeatures = 1000;
numSelectedFeatures = 100;
numPositiveExamples = 15;  % E.g., Autism
numNegativeExamples = 10;  % E.g., Typically developing subjects
numExamples = numPositiveExamples + numNegativeExamples;
labels = ones(numExamples, 1);
labels(1:numNegativeExamples) = -1;
features = randn(numExamples, numFeatures);

disp('Classify without feature selection')
% Cross validation. Leave one out
numCorrectlyClassified = 0;
for i = 1:numExamples
    idx = [1:i-1, i+1:numExamples]; % Leave out example i
    SVMStruct = fitcsvm(features(idx, :), labels(idx));
    predictedLabel(i)= predict(SVMStruct, features(i, :)); % Classify example i
    if (predictedLabel(i) == labels(i))
        numCorrectlyClassified = numCorrectlyClassified + 1;
    end
end

% Proportion of true results (both true positives and true negatives) among the total number of cases examined
accuracy = numCorrectlyClassified/numExamples;
disp(strcat('accuracy : ', num2str(accuracy)))
disp(' ')


% -------------PART C-------------------------------------------------------

disp('Classify with feature selection inside the cross validation')

% Cross validation. Leave one out
% numCorrectlyClassified = 0;
% r = zeros(numFeatures , 1);
%
% for i = 1:numFeatures
%     r(i) = similarityMeasure(features(:,i),labels);
%     % YOUR CODE GOES HERE
% end
% [rSorted,sortedFeatureIndices] = sort(r,'descend');
% selectedIndices = sortedFeatureIndices(1:numSelectedFeatures);
% sorted_features = features(:,selectedIndices);

% Proportion of true results (both true positives and true negatives) among the total number of cases examined
numCorrectlyClassified = 0;
r = zeros(numFeatures , 1);
for i = 1:numExamples
    idx = [1:i-1, i+1:numExamples]; % Leave out example i
    
    features_in = features(idx,:);
    labels_in = labels(idx);
    
    for j = 1:numFeatures
        r(j) = similarityMeasure(features_in(:,j),labels_in);
    end
    
    [rSorted,sortedFeatureIndices] = sort(r,'descend');
    selectedIndices = sortedFeatureIndices(1:numSelectedFeatures);
    sorted_features = features(:,selectedIndices);
    
    SVMStruct = fitcsvm(sorted_features(idx,:), labels(idx));
    predictedLabel(i)= predict(SVMStruct, sorted_features(i, :)); % Classify example i
    if (predictedLabel(i) == labels(i))
        numCorrectlyClassified = numCorrectlyClassified + 1;
    end
end

% Proportion of true results (both true positives and true negatives) among the total number of cases examined
accuracy = numCorrectlyClassified/numExamples;
disp(strcat('C: accuracy : ', num2str(accuracy)))
disp(' ')



% ------------PART D--------------------------------------------------------

disp('Classify with feature selection outside the cross validation')
% Feature selection
% Your code here
for j = 1:numFeatures
    r(j) = similarityMeasure(features(:,j),labels);
end

[rSorted,sortedFeatureIndices] = sort(r,'descend');
selectedIndices = sortedFeatureIndices(1:numSelectedFeatures);
sorted_features = features(:,selectedIndices);
% Cross validation. Leave one out
numCorrectlyClassified = 0;
for i = 1:numExamples
    idx = [1:i-1, i+1:numExamples]; % Leave out example i
    SVMStruct = fitcsvm(sorted_features(idx, :), labels(idx));
    predictedLabel(i)= predict(SVMStruct, sorted_features(i, :)); % Classify example i
    if (predictedLabel(i) == labels(i))
        numCorrectlyClassified = numCorrectlyClassified + 1;
    end
end

% Proportion of true results (both true positives and true negatives) among the total number of cases examined
accuracy = numCorrectlyClassified/numExamples;
disp(strcat('accuracy : ', num2str(accuracy)))
disp(' ')
