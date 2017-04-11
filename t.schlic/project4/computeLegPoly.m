function [ z ] = computeLegPoly( x, Q )
%COMPUTELEGPOLY Return the Qth order Legendre polynomial of x
%   Inputs:
%       x: vector (or scalar) of reals in [-1, 1]
%       Q: order of the Legendre polynomial to compute
%   Output:
%       z: matrix where each column is the Legendre polynomials of order 0 
%          to Q, evaluated atthe corresponding x value in the input

L=zeros(Q+1,size(x,2));

L(1,:)=1;
L(2,:)=x;
if Q>2
    for q=3:Q+1
        L(q,:)=(2*(q-1)-1)/(q-1)*x.*L(q-1,:)-(q-2)/(q-1)*L(q-2,:);
    end
end
z=L;

end


















