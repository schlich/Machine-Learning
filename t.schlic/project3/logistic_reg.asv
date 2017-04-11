function [ w e_in ] = logistic_reg( X, y, max_its )
%LOGISTIC_REG Learn logistic regression model using gradient descent
%   Inputs:
%       X : data matrix
%       Y : data labels (plus or minus 1)
%   Outputs:
%       w : weight vector
%       e_in : in-sample error (as defined in LFD)
    
    N=size(X,1);
    X0=ones(N,1);
    %add offset
    X=[X0 X];
    eta=1e-5;
    
    

    %initialize weight vector
    dim=size(X,2); %dimensionality of input space
    w_t=zeros(dim,1); 
    for i=1:max_its

        g=-(1/N).*sum(bsxfun(@times,y./(1+exp(y.*(X*w_t))),X));
        if norm(g)<.001
            break
        end
        v=-g;       %move in direction opposite gradient
        w_t=w_t + eta.*v' ;   %update weight vector
        
    end
    w=w_t;
    e_in=1/N*sum(log((1+exp(-y.*(X*w)))));
    
end

