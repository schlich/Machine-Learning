function [ train_set test_set ] = generate_dataset( Q_f, N_train, N_test, sigma )
%GENERATE_DATASET Generate training and test sets for the Legendre
%polynomials example
%   Inputs:
%       Q_f: order of the hypothesis
%       N_train: number of training examples
%       N_test: number of test examples
%       sigma: standard deviation of the stochastic noise
%   Outputs:
%       train_set and test_set are both 2-column matrices in which each row
%       represents an (x,y) pair
    
x_train=2*rand(1,N_train)-1;
z=computeLegPoly(x_train,Q_f);
epsilon=randn(1,N_train);

%apply normalization factor derived from part A
norm_factor=0;
for q=0:Q_f
    norm_factor=norm_factor + 1/(2*q+1);
end

a=1/sqrt(norm_factor).*(randn(1,Q_f+1))';

f=sum(bsxfun(@times,a,z));

y_train=f+sigma*epsilon;

train_set=[x_train' y_train'];

x_test=2*rand(1,N_test)-1;
z=computeLegPoly(x_test,Q_f);
epsilon=randn(1,N_test);

f=zeros(size(x_test));
f=sum(bsxfun(@times,a,z));


y_test=f+sigma*epsilon;

test_set=[x_test' y_test'];

    
end

