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
data_set = 21;
train_set = 15;
test_set = data_set - train_set;
lth = size(embeddings_th,1);
rank_expct = 1;
maxIters = 101;
times = 20;
opts.L = 128; % code length (in binary bits)
opts.K = 16; % options for subspace dimension K
opts.lambda = 0.2; % options for relative penalty
opts.beta = 2; % options for softness (alpha in the paper)
opts.pt1 = 0.5; % new loss 2 penalty
opts.pt2 = 0.5; % new loss 3 penalty
%% Test n times
for n = 1:times
    fprintf(' ## Train %d\n',n);
    % Initiate data  
    arr = sort(randperm(lth));
    split = sort(randperm(data_set,test_set));
    arr_te = [];
    for i = 0:data_set:lth-1
        arr_te = [arr_te split+i];
    end
    arr_tr = setdiff(arr,arr_te);
    Ytr = embeddings_th(arr_tr,:)';
    Xtr = embeddings_vis(arr_tr,:)';
    Yte = embeddings_th(arr_te,:)';
    Xte = embeddings_vis(arr_te,:)';
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
    % X, Y, S
    train_data.X = Xtr;
    train_data.Y = Ytr;
    Str = zeros(size(Xtr,2));
    for i = 0:train_set:size(Str,2)-1
        Str(i+1:i+train_set,i+1:i+train_set) = 1;
    end
    train_data.S = Str;       
    test_data.X = Xte;
    test_data.Y = Yte;
    % Training and encoding
    fprintf(' ## sph training, L = %d; K = %d; lambda = %d; beta = %d; pt1 = %d; pt2 = %d\n', opts.L, opts.K, opts.lambda, opts.beta, opts.pt1, opts.pt2);
    [~, model] = sph(train_data, opts, maxIters);
    [test_code, ~] = sph(test_data, opts, maxIters, model);
    Hxt = test_code.Hx; Hyt = test_code.Hy;
    %% Match
    % Test: vis vs th
    [test_tmp, ~] = performance_rn(Hxt,Hyt,test_set,rank_expct);
    test(n,1) = test_tmp;
    % Test: vis vs vis
    [test_tmp, ~] = performance_rn(Hxt,Hxt,test_set,rank_expct);
    test(n,2) = test_tmp;
    % Test: th vs th
    [test_tmp, ~] = performance_rn(Hyt,Hyt,test_set,rank_expct);
    test(n,3) = test_tmp;
    disp(test(n,:));
end
%% Calculate avg
test = mean(test,1);
fprintf(' ## Final Rank-1 rate: vis vs thermal %.4f;vis vs vis %.4f;thermal vs thermal %.4f\n', test(1,1), test(1,2), test(1,3));