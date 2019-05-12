%% Pattern Recognition 2019
%  Exercise 1.3 | Linear Discriminant Analysis
%
%  Instructions
%  ------------
%
%  This file contains code that helps you get started on the
%  exercise. You will need to complete the following functions:
%
%     myPCA.m
%     fisherLinearDiscriminant.m  (for 2-class LDA)
%	  myLDA.m  (for multiclass LDA)
%     projectDataLDA.m
%     projectDataPCA.m
%     recoverDataLDA.m
%     recoverDataPCA.m
%
%

%% Initialization
clear ; close all; clc

% ================== Part 1: Load the Dataset  ===================
%  We start using a small dataset that is easy to
%  visualize
%
fprintf('Visualizing example dataset for LDA.\n\n');

%  The following command loads the dataset. You should now have the 
%  variable X in your environment
load ('ex1_3_data1.mat');

%  Before running PCA, it is important to first normalize X
[X_norm, mu, sigma] = featureNormalize(X);

X1 = X_norm(c==1, :);
X2 = X_norm(c==2, :);

%  Visualize the example dataset
figure(1)
title('LDA')
hold on
plot(X1(:, 1), X1(:, 2), 'ro');
plot(X2(:, 1), X2(:, 2), 'ks');
axis([-2.1 2.1 -2.1 2.1]); axis square;
hold off


fprintf('Program paused. Press enter to continue.\n');
pause;


% =============== 2-Class Fisher Linear Discriminant =============
% Add your code to the implement the following function
%

v = fisherLinearDiscriminant(X1, X2);


hold on
drawLine(-5*v', 5*v', '-g', 'LineWidth', 1);
hold off

fprintf('Program paused. Press enter to continue.\n');
pause;

%  Project the data onto the direction of the one dimensional vector v
[Z1 ] = projectDataLDA(X1, v);
[Z2 ] = projectDataLDA(X2, v);




% Reconstruct the data on the line defined by vector v
[X1_rec]  = recoverDataLDA(Z1, v);
[X2_rec]  = recoverDataLDA(Z2, v);



%  Draw lines connecting the projected points to the original points
fprintf('\nDisplaying LDA on example dataset.\n\n');

hold on;
plot(X1_rec(:, 1), X1_rec(:, 2), 'bo');
hold off
hold on
for i = 1:size(X1, 1)
    drawLine(X1(i,:), X1_rec(i,:), '--k', 'LineWidth', 1);
end

hold off
hold on
plot(X2_rec(:, 1), X2_rec(:, 2), 'rs');
hold off
hold on
for i = 1:size(X2, 1)
    drawLine(X2(i,:), X2_rec(i,:), '--k', 'LineWidth', 1);
end

hold off
legend({'Normalized data of class 1 ','Normalized data of class 2 ','1st component','Recovered Data of class 1 ','Recovered Data of class 2','Projection lines'},'Location','northwest')
fprintf('Program paused. Press enter to continue.\n');
pause;



% =============== Principal Component Analysis ===============

fprintf('\nRunning PCA on example dataset.\n\n');

%  Run PCA
[U, S] = myPCA(X_norm);


%  Project the data onto K = 1 dimension
K = 1;
Z = projectDataPCA(X_norm, U, K);

X_rec  = recoverDataPCA(Z, U, K);

%  Plot the normalized dataset (returned from principalComponentAnalysis)
%  Draw lines connecting the projected points to the original points
fprintf('\nVisualizing example dataset for PCA.\n\n');
figure(2)
title('PCA')
hold on;
axis([-2.1 2.1 -2.1 2.1]); axis square;
drawLine(-2.3*U(:,1), 2.3*U(:,1), '-.r', 'LineWidth', 1);

plot(X1(:, 1), X1(:, 2), 'bo');
plot(X2(:, 1), X2(:, 2), 'rs');
plot(X_rec(c==1, 1), X_rec(c==1, 2), 'bo', 'MarkerFaceColor', 'b');
plot(X_rec(c==2, 1), X_rec(c==2, 2), 'rs', 'MarkerFaceColor', 'r');
for i = 1:size(X_norm, 1)
    drawLine(X_norm(i,:), X_rec(i,:), '--k', 'LineWidth', 1);
end
legend({'1st component','Normalized data of class 1 ','Normalized data of class 2 ','Recovered Data of class 1 ','Recovered Data of class 2','Projection lines'},'Location','northwest')
hold off


%% =============== PART B: FisherIris DATA ===============
% Apply LDA to the Fisher Iris Dataset
clear ; close all; clc
%Load Fisher Iris Data

load('fisheriris.mat');

% Convert the species cell into an array containig class labels
% Class 0 for "setosa"
% Class 1 for "versicolor"
% Class 2 for "virginica"
iris_labels = 1*cellfun(@(x)isequal(x,'versicolor'),species)+2*cellfun(@(x)isequal(x,'virginica'),species);

%  Before running PCA, it is important to first normalize X
[meas_norm, mu, sigma] = featureNormalize(meas);

% Get the data for each class
IRIS1 = meas_norm(1:50,:);    %Samples of Class 0
IRIS2 = meas_norm(51:100,:);  %Samples of Class 1
IRIS3 = meas_norm(101:end,:); %Samples of Class 2


%  Visualize the example dataset
figure(3)
hold on
plot(IRIS1(:, 1), IRIS1(:, 2), 'bo');
plot(IRIS2(:, 1), IRIS2(:, 2), 'rs');
plot(IRIS3(:, 1), IRIS3(:, 2), 'g+');
title('IRIS - Original data');
legend({'setosa','versicolor','Virginica'},'Location','northwest')

hold off

NewDim = 2; %The new feature dimension after applying LDA
v = myLDA(meas_norm, iris_labels, NewDim);

%  Project the data on the direction of the two dimensional v
[meas_reduced ] = projectDataLDA(meas_norm, v)';

%  Visualize the sample dataset after LDA is applied
%  Use different color/symbol for each class
figure(4)

setosa = meas_reduced(1:50,:);    %Samples of Class 0
versicolor = meas_reduced(51:100,:);  %Samples of Class 1
Virginica = meas_reduced(101:end,:); %Samples of Class 2


%  Visualize the example dataset
hold on
title('IRIS - Classified data');

plot(setosa(:, 1), setosa(:, 2), 'bo');
plot(versicolor(:, 1), versicolor(:, 2), 'rs');
plot(Virginica(:, 1), Virginica(:, 2), 'g+');
legend({'setosa','versicolor','Virginica'},'Location','northwest')
hold off

%%
