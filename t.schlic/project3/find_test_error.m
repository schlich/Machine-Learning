function [ test_error ] = find_test_error( w, X, y )
%FIND_TEST_ERROR Find the test error of a linear separator
%   This function takes as inputs the weight vector representing a linear
%   separator (w), the test examples in matrix form with each row
%   representing an example (X), and the labels for the test data as a
%   column vector (y). The labels are assumed to be plus or minus one. The
%   function returns the error on the test examples as a fraction. The
%   hypothesis is assumed to be of the form (sign ( [1 x(n,:)] * w )

sum=0; %running sum

N=size(X,1);
for n=1:size(X,1)
    if sign([1 X(n,:)]*w)==y(n)
        sum=sum+1;
    end
end
test_error=1-sum/N;

end

