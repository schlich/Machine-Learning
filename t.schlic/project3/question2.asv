test=load('zip.test');
train=load('zip.train');

y=train(:,1); 
X=train(:,2:end);

y(y~=1)=0;
w=glmfit(X,y,'binomial');

X0=ones(size(X,1),1);
X=[X0 X];

y_star=sign(X*w);
y_star(y_star==-1)=0;

zip_train_error=1-sum(y_star==y)/size(y,1)

%classify test data
y=test(:,1);
X=test(:,2:end);
y(y~=1)=0;

X0=ones(size(X,1),1);
X=[X0 X];

y_star=sign(X*w);
y_star(y_star==-1)=0;

zip_test_error=1-sum(y_star==y)/size(y,1)

clear all
test=load('features.test');
train=load('features.train');

y=train(:,1); 
X=train(:,2:end);

y(y~=1)=0;
w=glmfit(X,y,'binomial');

X0=ones(size(X,1),1);
X=[X0 X];

y_star=sign(X*w);
y_star(y_star==-1)=0;

feat_train_error=1-sum(y_star==y)/size(y,1)

%classify test data
y=test(:,1);
X=test(:,2:end);
y(y~=1)=0;

X0=ones(size(X,1),1);
X=[X0 X];

y_star=sign(X*w);
y_star(y_star==-1)=0;

feat_test_error=1-sum(y_star==y)/size(y,1)