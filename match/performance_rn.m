function [rec_rank, rec_rates]= performance_rn(featuresA,featuresB,features_set,rank_expct)
% Rank-N performance
A = featuresA;
B = featuresB;
n = size(A,1);
class = size(A,2)/features_set;
A_ids = [];
B_ids = [];
for i=1:class
    for j=1:features_set
        A_ids = [A_ids i];
        B_ids = [B_ids,i];
    end
end
results = nn_classification_PhD(A, A_ids, B, B_ids, n, 'cos', 'all');
[rec_rates, ranks] = produce_CMC_PhD(results);
rec_rank = rec_rates(1,rank_expct);