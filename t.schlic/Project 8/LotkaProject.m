close all
clear all
clc

tinitial=0;
tfinal=30;

%change inital u and v values here ([u v])
y0=[10 30];
[t y]=ode45('LotkaM',[tinitial tfinal],y0);


figure
plot(t,y(:,1),t,y(:,2),'r')
title('Population v Time')
xlabel('Time')
ylabel('Population')
legend('Prey','Predator')