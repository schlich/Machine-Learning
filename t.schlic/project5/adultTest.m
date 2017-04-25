
%code to convert data into numerical table format borrowed from
%
%%
%extract training data
adult = urlread(['http://archive.ics.uci.edu/ml/'...
'machine-learning-databases/adult/adult.data']);
VarNames = {'age' 'workclass' 'fnlwgt' 'education' 'educationNum'...
'maritalStatus' 'occupation' 'relationship' 'race'...
'sex' 'capitalGain' 'capitalLoss'...
'hoursPerWeek' 'nativeCountry' 'income'};
adult = strrep(adult,'?','NaN');
adult = textscan(adult,'%f%s%f%s%f%s%s%s%s%s%f%f%f%s%s',...
'Delimiter',',');
adult = table(adult{:},'VariableNames',VarNames);
X = classreg.regr.modelutils.predictormatrix(adult,'ResponseVar',...
size(adult,2));
Y = nominal(adult.income);
%%
%calculate tree
tree=fitctree(X,Y,'CategoricalPredictors',logical([0 1 0 1 0 1 1 1 1 1 0 0 0 1]));
%find level that minimizes cross validation error
[~,~,~,bestlevel] = cvLoss(tree,'SubTrees','All','TreeSize','min');
%prune tree to that level
tree2=prune(tree,'Level',bestlevel);
%%
%extract test data
adulttest = urlread(['http://archive.ics.uci.edu/ml/'...
'machine-learning-databases/adult/adult.test']);
adulttest = adulttest(21:end);
adulttest = strrep(adulttest,'?','NaN');
adulttest = strrep(adulttest,'.','');
adulttest = textscan(adulttest,'%f%s%f%s%f%s%s%s%s%s%f%f%f%s%s',...
'Delimiter',',');
adulttest = table(adulttest{:},'VariableNames',VarNames);
Xtest = classreg.regr.modelutils.predictormatrix(adulttest,'ResponseVar',...
size(adulttest,2));
Ytest = nominal(adulttest.income);
%%
%calculate test error on original tree and pruned tree
err=sum(Ytest~=predict(tree,Xtest))/max(size(Ytest));
prunerr=sum(Ytest~=predict(tree2,Xtest))/max(size(Ytest));