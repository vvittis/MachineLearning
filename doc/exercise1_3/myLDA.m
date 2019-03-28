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
    
%     mu1(1,:) = mean(Samples(1:50,:))
    %Calculate the Class Mean 
    
     for i = 1:NumClasses
         step = ((i-1)*50);
         mu(i,:) = mean(Samples(step+1:step+50,:));
    end
	mu;
    %Calculate the Within Class Scatter Matrix
    Sw = zeros();

    for i = 1:NumClasses
         step = ((i-1)*50);
         P(i)* cov(Samples(step+1:step+50,:));
         Sw = Sw + P(i)* cov(Samples(step+1:step+50,:));
    end
    Sw;
    %Calculate the Global Mean
	 for i = 1:NumFeatures
         m0(i) = 0;
         for j = 1:NumClasses
             m0(i) = P(j).*( m0(i) + mu(j,i));
         end 
     end
     m0;
     
    %Calculate the Between Class Scatter Matrix
    Sb = zeros();
    
	 for i = 1:NumClasses
         Sb1(i,:) = mu(i,:) - m0;
         Sb = Sb  + P(i)*((mu(i,:) - m0)*(mu(i,:) - m0)');
     end
     Sb ;
    
    %Eigen matrix EigMat=inv(Sw)*Sb
     EigMat = inv(Sw)*Sb;
    
    %Perform Eigendecomposition
     [U,S]          = eig(EigMat)
     eigenval       = diag(S);
     [eigenval,ind] = sort(eigenval,1,'descend'); %Sort them 
      U             = U(:,ind) %Corresponding eigenvectors
      %Select the NewDim eigenvectors corresponding to the top NewDim
      S             = diag(eigenval);
      
    
    %eigenvalues (Assuming they are NewDim<=NumClasses-1)
	%% You need to return the following variable correctly.
	A=zeros(NumFeatures,NewDim); % Return the LDA projection vectors
    A             = U(:,1:NewDim) ;
