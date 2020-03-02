clear all;
addpath('matlab_tools');
addpath('PhD_tool');
addpath_recurse("PhD_tool");
addpath('sph');
addpath('match');
close all;
%% Load data and settings
load('data/embeddings_th.mat');
load('data/embeddings_vis.mat');
group = 50;
group_npe = 10;
group_pre = group - group_npe;
train_set = 15;
test_set = 6;
match_set = 21;
lth = size(embeddings_th,1);
rank_expct = 1;
maxIters = 101;
times = 2;
opts.L = 2;
opts.K = 16;
opts.lambda = 0.2; % options for relative penalty
opts.beta = 2; % options for softness (alpha in the paper)
opts.pt1 = 0.5;
opts.pt2 = 0.5;
expression_set = [1:test_set];
offset = 0;
default_set = test_set;
%% Test n times
for n = 1:times
    fprintf(' ## Train %d\n',n);
    % Initiate data 
    arr_group = sort(randperm(group));
    arr_pre = sort(randperm(group,group_pre));
    arr_npe = setdiff(arr_group,arr_pre);
    arr_match_set = sort(randperm(match_set));
    arr_train_set = sort(randperm(match_set-1,train_set-1));
    arr_train_set = [1 arr_train_set+1];
    arr_test_set = setdiff(arr_match_set,arr_train_set);
    arr_tr_pre = [];
    for i = arr_pre
        arr_tr_pre = [arr_tr_pre arr_train_set+(i-1)*match_set];
    end
    arr_te_pre = [];
    for i = arr_pre
        arr_te_pre = [arr_te_pre arr_test_set+(i-1)*match_set];
    end
    arr_tr_npe = [];
    for i = arr_npe
        arr_tr_npe = [arr_tr_npe arr_train_set+(i-1)*match_set];
    end
    arr_te_npe = [];
    for i = arr_npe
        arr_te_npe = [arr_te_npe arr_test_set+(i-1)*match_set];
    end
    Ytr_pre = embeddings_th(arr_tr_pre,:)';
    Xtr_pre = embeddings_vis(arr_tr_pre,:)';
    Yte_pre = embeddings_th(arr_te_pre,:)';
    Xte_pre = embeddings_vis(arr_te_pre,:)';
    Ytr_npe = embeddings_th(arr_tr_npe,:)';
    Xtr_npe = embeddings_vis(arr_tr_npe,:)';
    Yte_npe = embeddings_th(arr_te_npe,:)';
    Xte_npe = embeddings_vis(arr_te_npe,:)';
    Ytr = [Ytr_pre Ytr_npe];
    Xtr = [Xtr_pre Xtr_npe];
    Yte = [Yte_pre Yte_npe];
    Xte = [Xte_pre Xte_npe];
    % center, then normalize data        
    mX = mean(Xtr_pre,2);
    Xtr_pre = bsxfun(@minus, Xtr_pre, mX);
    normXtr_pre = sqrt(sum(Xtr_pre.^2, 1));
    Xtr_pre = bsxfun(@rdivide, Xtr_pre, normXtr_pre);
    Xte_pre = bsxfun(@minus, Xte_pre, mX);
    normXte_pre = sqrt(sum(Xte_pre.^2, 1));
    Xte_pre = bsxfun(@rdivide, Xte_pre, normXte_pre);
    mX = mean(Xtr_npe,2);
    Xtr_npe = bsxfun(@minus, Xtr_npe, mX);
    normXtr_npe = sqrt(sum(Xtr_npe.^2, 1));
    Xtr_npe = bsxfun(@rdivide, Xtr_npe, normXtr_npe);
    Xte_npe = bsxfun(@minus, Xte_npe, mX);
    normXte_npe = sqrt(sum(Xte_npe.^2, 1));
    Xte_npe = bsxfun(@rdivide, Xte_npe, normXte_npe);
    mY = mean(Ytr_pre,2);
    Ytr_pre = bsxfun(@minus, Ytr_pre, mY);
    normYtr_pre = sqrt(sum(Ytr_pre.^2, 1));
    Ytr_pre = bsxfun(@rdivide, Ytr_pre, normYtr_pre);
    Yte_pre = bsxfun(@minus, Yte_pre, mY);
    normYte_pre = sqrt(sum(Yte_pre.^2, 1));
    Yte_pre = bsxfun(@rdivide, Yte_pre, normYte_pre);
    mY = mean(Ytr_npe,2);
    Ytr_npe = bsxfun(@minus, Ytr_npe, mY);
    normYtr_npe = sqrt(sum(Ytr_npe.^2, 1));
    Ytr_npe = bsxfun(@rdivide, Ytr_npe, normYtr_npe);
    Yte_npe = bsxfun(@minus, Yte_npe, mY);
    normYte_npe = sqrt(sum(Yte_npe.^2, 1));
    Yte_npe = bsxfun(@rdivide, Yte_npe, normYte_npe);
    % X, Y, S
    train_data_pre.X = Xtr_pre;
    train_data_pre.Y = Ytr_pre;
    Str_pre = zeros(size(Xtr_pre,2));
    Ctr_pre = [];
    for i = 0:train_set:size(Str_pre,2)-1
        Str_pre(i+1:i+train_set,i+1:i+train_set) = 1;
        Ctr_pre = [Ctr_pre i+1];
    end
    train_data_pre.S = Str_pre;
    train_data_pre.C = Ctr_pre;
    test_data_pre.X = Xte_pre;
    test_data_pre.Y = Yte_pre;   
    train_data_npe.X = Xtr_npe;
    train_data_npe.Y = Ytr_npe;
    Str_npe = zeros(size(Xtr_npe,2));
    Ctr_npe = [];
    for i = 0:train_set:size(Str_npe,2)-1
        Str_npe(i+1:i+train_set,i+1:i+train_set) = 1;
        Ctr_npe = [Ctr_npe i+1];
    end
    train_data_npe.S = Str_npe;
    train_data_npe.C = Ctr_npe;
    test_data_npe.X = Xte_npe;
    test_data_npe.Y = Yte_npe;   
    % Train pre model
    pre_model = [];
    fprintf(' ## Train %d - Pre-train model\n',n);
    [~, pre_model] = sph_il(train_data_pre, opts, maxIters, pre_model);
    % Train new model and encode
    fprintf(' ## Train %d - Train new model\n',n);
    [~, model] = sph_il(train_data_npe, opts, maxIters, pre_model);
    [test_code_pre, ~] = sph_il(test_data_pre, opts, maxIters, pre_model, model);
    Hxt_pre = test_code_pre.Hx'; Hyt_pre = test_code_pre.Hy';
    [test_code_npe, ~] = sph_il(test_data_npe, opts, maxIters, pre_model, model);
    Hxt_npe = test_code_npe.Hx'; Hyt_npe = test_code_npe.Hy';
    Hxt = [Hxt_pre;Hxt_npe];Hyt = [Hyt_pre;Hyt_npe];
    %% sph-il match
    dist_type = 'jaccard';
    [EER] = performance_ver(Hxt,Hyt,expression_set,offset,default_set,dist_type);
    test(n,1) = mean(EER(:));
    [EER] = performance_ver(Hxt,Hxt,expression_set,offset,default_set,dist_type);
    test(n,2) = mean(EER(:));
    [EER] = performance_ver(Hyt,Hyt,expression_set,offset,default_set,dist_type);
    test(n,3) = mean(EER(:));
    disp(test(n,:));
    %% All train
    % center, then normalize data
    mX = mean(Xtr,2);
    Xtr = bsxfun(@minus, Xtr, mX);
    normX = sqrt(sum(Xtr.^2, 1));
    Xtr = bsxfun(@rdivide, Xtr, normX);
    Xte = bsxfun(@minus, Xte, mX);
    normXte = sqrt(sum(Xte.^2, 1));
    Xte = bsxfun(@rdivide, Xte, normXte);
    mY = mean(Ytr,2);
    Ytr = bsxfun(@minus, Ytr, mY);
    normY = sqrt(sum(Ytr.^2, 1));
    Ytr = bsxfun(@rdivide, Ytr, normY);
    Yte = bsxfun(@minus, Yte, mY);
    normYte = sqrt(sum(Yte.^2, 1));
    Yte = bsxfun(@rdivide, Yte, normYte);
    train_data.X = Xtr;
    train_data.Y = Ytr;
    Str = zeros(size(Xtr,2));
    for i = 0:train_set:size(Str,2)-1
        Str(i+1:i+train_set,i+1:i+train_set) = 1;
    end
    train_data.S = Str;
    test_data.X = Xte;
    test_data.Y = Yte;
    % Train and hash
    fprintf(' ## Train %d - sph train\n',n);
    [~, model] = sph(train_data, opts, maxIters);
    [test_code, ~] = sph(test_data, opts, maxIters, model);
    Hxt = test_code.Hx'; Hyt = test_code.Hy';
    %% Match all train
    dist_type = 'jaccard';
    % sph match
    [EER] = performance_ver(Hxt,Hyt,expression_set,offset,default_set,dist_type);
    test_all(n,1) = mean(EER(:));
    [EER] = performance_ver(Hxt,Hxt,expression_set,offset,default_set,dist_type);
    test_all(n,2) = mean(EER(:));
    [EER] = performance_ver(Hyt,Hyt,expression_set,offset,default_set,dist_type);
    test_all(n,3) = mean(EER(:));
    disp(test_all(n,:));
end
%% Avg
test_avg = mean(test,1);
fprintf(' ## Final SPH-IL EER: vis vs thermal %.4f;vis vs vis %.4f;thermal vs thermal %.4f\n', test_avg(1,1), test_avg(1,2), test_avg(1,3));
test_all_avg = mean(test_all,1);
fprintf(' ## Final SPH EER: vis vs thermal %.4f;vis vs vis %.4f;thermal vs thermal %.4f\n', test_all_avg(1,1), test_all_avg(1,2), test_all_avg(1,3));