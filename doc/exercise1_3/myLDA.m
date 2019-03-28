function A = myLDA(Samples, Labels, NewDim)
% Input:    
%   Samples: The Data Samples 
%   Labels: The labels that correspond to the Samples
%   NewDim: The New Dimension of the Feature Vector after applying LDA

	[NumSamples NumFeatures] = size(Samples);
    
    A=zeros(NumFeatures,NewDim);
    
    NumLabels = length(Labels);
    if(NumSamples ~= NumLabels) then
        fprintf('\nNumber of Samples are not the same with the Number of Labels.\n\n');
        exit
    end
    Classes = unique(Labels);
    NumClasses = length(Classes);  %The number of classes

    %For each class i
	%Find the necessary statistics
    
    %Calculate the Class Prior Probability
    for i = 1:NumClasses
        P(i)= sum(Labels()==(i-1))/size(Labels,1);
    end
    
    mu(1,:) = mean(Samples(1:50,:))
    %Calculate the Class Mean 
%      for i = 1:NumClasses
%        mu(i,:) = mean(Samples());
%     end
	
    %Calculate the Within Class Scatter Matrix
% 	Sw=
    %Calculate the Global Mean
% 	m0=

  
    %Calculate the Between Class Scatter Matrix
% 	Sb= 
    
    %Eigen matrix EigMat=inv(Sw)*Sb
%     EigMat = inv(Sw)*Sb;
    
    %Perform Eigendecomposition

    
    %Select the NewDim eigenvectors corresponding to the top NewDim
    %eigenvalues (Assuming they are NewDim<=NumClasses-1)
	%% You need to return the following variable correctly.
% 	A=zeros(NumFeatures,NewDim);  % Return the LDA projection vectors
