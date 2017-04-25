function [ oobErr ] = BaggedTrees( X, Y, numBags,X_test,Y_test )
%BAGGEDTREES Returns out-of-bag classification error of an ensemble of
%numBags CART decision trees on the input dataset, and also plots the error
%as a function of the number of bags from 1 to numBags
%   Inputs:
%       X : Matrix of training data
%       Y : Vector of classes of the training examples
%       numBags : Number of trees to learn in the ensemble
%
%   You may use "fitctree" but do not use "TreeBagger" or any other inbuilt
%   bagging function

warning off

oobErrs=NaN(numBags,1);

[N,d]=size(X);
votes=NaN(max(size(Y_test)),numBags);
consensus=zeros(N,1);

for j=1:numBags
    bag=zeros(N,d);
    bagLabels=zeros(N,1);
    for i=1:N
        n=randi(N);             %choose a random sample
        bag(i,:)=X(n,:);        %put it in the bag
        bagLabels(i,:)=Y(n);    %keep track of labels
    end

    tree=fitctree(bag,bagLabels);
    predictions=predict(tree,X_test);
    votes(:,j)=predictions;
    consensus=mode(votes');
    oobErrs(j)=1-sum(consensus'==Y_test)/max(size(Y_test));    
    
end

figure
plot(oobErrs)
oobErr=oobErrs(end);

end


