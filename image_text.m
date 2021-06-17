clear all;
addpath('sph');
addpath('match');
close all;
%% Settings
maxIters = 101;
times = 20; % random time
opts.L = 128; % bits length
opts.K = 8; % options for subspace dimension K
opts.lambda = 1; % options for relative penalty
opts.beta = 8; % options for softness (alpha in the paper)
opts.pt1 = 0; % new loss 1 penalty
opts.pt2 = 0.5; % new loss 2 penalty
%% Load and preprocess data
dat = load('data/raw_features.mat');
ttlabels = textscan(fopen('data/testset_txt_img_cat.list'), '%s %s %u8'); 
tnlabels = textscan(fopen('data/trainset_txt_img_cat.list'), '%s %s %u8'); 
Ytr = dat.I_tr';
Xtr = dat.T_tfidf_tr';
Yte = dat.I_te';
Xte = dat.T_tfidf_te';
ttlabels = ttlabels{1, 3};
tnlabels = tnlabels{1, 3};
% center, then normalize data
mX = mean(Xtr,2);
Xtr = bsxfun(@minus, Xtr, mX);
normX = sqrt(sum(Xtr.^2, 1));
Xtr = bsxfun(@rdivide, Xtr, normX);       
Xte = bsxfun(@minus, Xte, mX);
normXte = sqrt(sum(Xte.^2, 1));
Xte = bsxfun(@rdivide, Xte, normXte);
mY = mean(Ytr,2);
Ytr = bsxfun(@minus, Ytr, mean(Ytr,2));
normY = sqrt(sum(Ytr.^2, 1));
Ytr = bsxfun(@rdivide, Ytr, normY);
Yte = bsxfun(@minus, Yte, mY);
normYte = sqrt(sum(Yte.^2, 1));
Yte = bsxfun(@rdivide, Yte, normYte);    
% X, Y, S
train_data.X = Xtr;
train_data.Y = Ytr;
query_data.X = Xte;
query_data.Y = Yte;
train_data.S = uint8(bsxfun(@eq, tnlabels, tnlabels'));
query_data.S = uint8(bsxfun(@eq, ttlabels, tnlabels'));
%% Tarin and match
test = [];
%% Test n times
for n = 1:times
    fprintf(' ## Train %d\n',n);
    % Training and encoding
    fprintf(' ## sph training, L = %d; K = %d; lambda = %d; beta = %d; pt1 = %d\n; pt2 = %d\n', opts.L, opts.K, opts.lambda, opts.beta, opts.pt1, opts.pt2);
    [~, model] = sph(train_data, opts, maxIters);       
	[test_code, ~] = sph(query_data, opts, maxIters, model);        
	[db_code, ~] = sph(train_data, opts, maxIters, model);
	Hxt = test_code.Hx; Hyt = test_code.Hy;
	Hxdb = db_code.Hx; Hydb = db_code.Hy;
	% Compute mean average precision
	fprintf(' ## sph computing mAP ...\n');   
	DxyTestDb = pdist2(Hxt', Hydb', 'hamming')*opts.L;
	[~, mAPxy, ~, ~, ~] = test_hash(DxyTestDb, query_data.S, opts.L, []);           
	DyxTestDb = pdist2(Hyt', Hxdb', 'hamming')*opts.L;
	[~, mAPyx, ~, ~, ~] = test_hash(DyxTestDb, query_data.S, opts.L, []);
	fprintf(' ## sph DONE. mAPxy %.4f/mAPyx %.4f.\n', mAPxy, mAPyx);
	test = [test [mAPxy;mAPyx]];
end
%% Avg
test_final = mean(test,2);
fprintf(' ## Final mAPxy %.4f/mAPyx %.4f.\n', test_final(1,1), test_final(2,1));