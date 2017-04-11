function LV=LotkaM(t,y)

%change k, a values here
k=10;
a=.5;

LV(1)=y(1)-(y(1)-k)*y(2);
LV(2)=a*y(2)*(y(1)-k-1);
LV=[LV(1) LV(2)]';

end

