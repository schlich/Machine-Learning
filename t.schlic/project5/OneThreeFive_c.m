% Script to load data from zip.train, filter it into datasets with only one
% and three or three and five, and compare the performance of plain
% decision trees (cross-validated) and bagged ensembles (OOB error)
load zip.train;
train=zip;
load zip.test;
test=zip;

fprintf('Working on the one-vs-five problem...\n\n');
subsample_train = train(find(train(:,1)==1 | train(:,1) == 5),:);
Y_train = subsample_train(:,1);
X_train = subsample_train(:,2:257);
subsample_test = test(find(test(:,1)==1 | test(:,1) == 5),:);
Y_test = subsample_test(:,1);
X_test = subsample_test(:,2:257);

ct = fitctree(X_train,Y_train);
Y_test_star=predict(ct,X_test);
testErr_1_5=1-sum(Y_test_star==Y_test)/max(size(Y_test));

fprintf('The test error for one tree is %.4f\n', testErr_1_5);
bee = BaggedTrees_c(X_train, Y_train, 200, X_test,Y_test);
fprintf('The test error for 200 bagged decision trees is %.4f\n', bee);


fprintf('\nNow working on the three-vs-five problem...\n\n');
subsample_train = train(find(train(:,1)==3 | train(:,1) == 5),:);
Y_train = subsample_train(:,1);
X_train = subsample_train(:,2:257);
subsample_test = test(find(test(:,1)==3 | test(:,1) == 5),:);
Y_test = subsample_test(:,1);
X_test = subsample_test(:,2:257);

ct = fitctree(X_train,Y_train);
Y_test_star=predict(ct,X_test);
testErr_3_5=1-(sum(Y_test_star==Y_test)/max(size(Y_test)));

fprintf('The test error for one tree is %.4f\n', testErr_3_5);
bee = BaggedTrees_c(X_train, Y_train, 200, X_test,Y_test);
fprintf('The test error for 200 bagged decision trees is %.4f\n', bee);