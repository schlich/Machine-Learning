function [ oobErr ] = BaggedTrees( X, Y, numBags )
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

oobErrs=zeros(numBags,1);

[N,d]=size(X);
votes=cell(N,1);
consensus=zeros(N,1);

for j=1:numBags
    bag=zeros(N,d);
    bagLabels=zeros(N,1);
    for i=1:N
    n=randi(N);             %choose a random sample
    bag(i,:)=X(n,:);        %put it in the bag
    bagLabels(i,:)=Y(n);    %keep track of labels
    end
    
    [oob_set,oob_i]=setdiff(X,bag,'rows');  %collect out-of-bag samples

    tree=fitctree(bag,bagLabels);
    predictions=predict(tree,oob_set);

    for i=1:size(predictions,1);
        votes{oob_i(i)}=[votes{oob_i(i)} predictions(i)];
    end
    
    for i=1:N
    consensus(i)=mode(votes{i});
    end
    oobErrs(j)=1-sum(consensus==Y)/N;    
    
end

figure
plot(oobErrs)
oobErr=oobErrs(end);

end


