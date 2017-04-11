test=load('features.test');
train=load('features.train');

X=train(:,2:end);
y=train(:,1); 
y(y~=1)=0;
w=glmfit(X,y,'binomial');

X0=ones(size(X,1),1);
X=[X0 X];

y_star=sign(X*w);
y_star(y_star==-1)=0;

feat_train_error=1-sum(y_star==y)/size(y,1)

%classify test data

y_test=test(:,1); 
y_test(y_test~=1)=0;
X_test=test(:,2:end);
X0_test=ones(size(X_test,1),1);
X_test=[X0_test test(:,2:end)];

y_test_star=sign(X_test*w);
y_test_star(y_test_star==-1)=0;

feat_test_error=1-sum(y_test_star==y_test)/size(y_star,1)