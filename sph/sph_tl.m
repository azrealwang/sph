function [code, model] = sph_tl(data, opts, maxIters, pre_model, model)
% If no model is provided, learn the model first
if nargin == 4
    model = trainCMRSH(data, opts, maxIters, pre_model);
end
% Do hash encoding
K = opts.K;
L = opts.L;
N = size(data.X, 2);
if ~isempty(data.X)
    Zx = model.Wx'*data.X; Zx = reshape(Zx, K, L*N);
    [~, Hx] = max(Zx); 
    Hx = reshape(Hx, L, N);
end
if ~isempty(data.Y)
    Zy = model.Wy'*data.Y; Zy = reshape(Zy, K, L*N);
    [~, Hy] = max(Zy);
    Hy = reshape(Hy, L, N);
end
code.Hx = Hx;
code.Hy = Hy;
%%=========================================================================
function model = trainCMRSH(data, opts, maxIters, pre_model)
X = data.X;
Y = data.Y;
S = data.S;
C = data.C;
ratio = 10000;
nb = 5;
if ~isempty(pre_model)
    model.X = [pre_model.X X(:,C)];
    model.Y = [pre_model.Y Y(:,C)];
    X = [pre_model.X X];
    Y = [pre_model.Y Y];
    S_tmp = zeros(size(S,2)+size(pre_model.X,2));
    for i = 1:size(pre_model.X,2)
        S_tmp(i,i) = 1;
    end
    for i = 1:size(S,1)
        for j = 1:size(S,2)
            S_tmp(i+size(pre_model.X,2),j+size(pre_model.X,2)) = S(i,j);
        end
    end
    S = S_tmp;
    ratio = 0.5*sqrt(size(C,2)/(size(C,2)+size(pre_model.X,2)));
    nb = 1;
else
    model.X = X(:,C);
    model.Y = Y(:,C);
end

K = opts.K;
L = opts.L;
lambda = opts.lambda;
beta = opts.beta;
pt1 = opts.pt1;
pt2 = opts.pt2;

S = double(S);
[dx, N] = size(X);
dy = size(Y, 1);

% training
bSize = min(500, N); % batch size
delta = eps; % stopping condition
momentum = 0.0; % weight update momentum

mu = 0.0;
Px = cell(1, L);
Py = cell(1, L);
alpha = ones(1, L)/L;
F = lambda-(lambda+1)*S;
A = ones(N); % Initial weights of every pair are 1
totalTime = 0;
DEBUG = 0;
sizePos = sum(S(:));
Npairs = lambda*(N*N-sizePos)+sizePos;

for n = 1 : L
    fprintf('Learn bit %d.\n', n); tStart = tic;    
    if ~isempty(pre_model)
        Wx = pre_model.Wx(:,(n-1)*K+1:n*K);
        Wy = pre_model.Wy(:,(n-1)*K+1:n*K);
    else
        Wx = normc((mvnrnd(zeros(dx, 1), diag(ones(dx, 1)), K))');
        Wy = normc((mvnrnd(zeros(dy, 1), diag(ones(dy, 1)), K))');
    end
    D = F.*A;
    Np = (sum(S(:).*A(:)))/Npairs;
    
    if mod(n, 4) == 0
        A = ones(N);
    end
    
    err = zeros(maxIters+1, 1);
    gnorm = zeros(maxIters, 1);
    evar = zeros(maxIters, 1);
    rate = zeros(maxIters, 1);
    
    % profile initialization error    
    Xe = exp(beta*bsxfun(@minus, Wx'*X, max(Wx'*X)));
    Hx = bsxfun(@rdivide, Xe, sum(Xe));    
    Ye = exp(beta*bsxfun(@minus, Wy'*Y, max(Wy'*Y)));
    Hy = bsxfun(@rdivide, Ye, sum(Ye));
    
    err1(1) = trace(Hx*D*Hy')/Npairs+Np;    
    err2(1) = 0;
    if pt1 > 0
        err2(1) = (trace(Hx*D*Hx')/Npairs+trace(Hy*D*Hy')/Npairs)/2+Np;
    end
    err3(1) = 0;
    if pt2 > 0
        dist = pdist([Hx Hy]','cosine')/2;
        err3(1) = 1-mean(dist(:));      
    end
    err(1) = (err1(1)+pt1*err2(1)+pt2*err3(1))/(1+pt1+pt2);
    
    dWx = 0;
    dWy = 0;
    eta = 4000.0; % learing rate    
    for i = 2 : maxIters                                           
        % randomly sample a batch
        bid = randperm(N, bSize);
        Xb = X(:, bid);
        Yb = Y(:, bid);
        Zxb = Wx'*Xb;
        Zyb = Wy'*Yb;
        Xeb = exp(beta*bsxfun(@minus, Zxb, max(Zxb)));
        Hxb = bsxfun(@rdivide, Xeb, sum(Xeb));    
        Yeb = exp(beta*bsxfun(@minus, Zyb, max(Zyb)));
        Hyb = bsxfun(@rdivide, Yeb, sum(Yeb));  
        
        Db = D(bid, bid);
        Hsxb = Hxb*Db;
        Hsyb = Hyb*Db';
                
        dWx_old = dWx;
        dWy_old = dWy;              
        dWx = Xb*((Hxb.*Hsyb)'-bsxfun(@times, diag(Hsyb'*Hxb), Hxb'))/(bSize*bSize);
        dWy = Yb*((Hyb.*Hsxb)'-bsxfun(@times, diag(Hsxb'*Hyb), Hyb'))/(bSize*bSize); 
        dWx = eta*dWx + momentum*dWx_old;
        dWy = eta*dWy + momentum*dWy_old;
        Wx = Wx - dWx;
        Wy = Wy - dWy;                               
        
        gnorm(i-1) = mean(abs(dWx(:)));
        if mod(i-1, nb) == 0
            Zx = Wx'*X;
            Zy = Wy'*Y;
            Xe = exp(beta*bsxfun(@minus, Zx, max(Zx)));
            Hx = bsxfun(@rdivide, Xe, sum(Xe));    
            Ye = exp(beta*bsxfun(@minus, Zy, max(Zy)));
            Hy = bsxfun(@rdivide, Ye, sum(Ye));  
            epoc = (i-1)/nb+1;
            % profile initialization error
            err1(epoc) = trace(Hx*D*Hy')/Npairs+Np;
            err2(epoc) = 0;
            if pt1 > 0
                err2(epoc) = (trace(Hx*D*Hx')/Npairs+trace(Hy*D*Hy')/Npairs)/2+Np;
            end
            err3(epoc) = 0;
            if pt2 > 0
                dist = pdist([Hx Hy]','cosine')/2;
                err3(epoc) = 1-mean(dist(:));
            end
            err(epoc) = (err1(epoc)+pt1*err2(epoc)+pt2*err3(epoc))/(1+pt1+pt2);
            
            if epoc >= 2
                % change learning rate adaptively
                if err(epoc) > err(epoc-1)
                    eta = eta * 0.5;
                else
                    eta = eta * 1.05; 
                end                       
                evar(epoc-1) = abs(err(epoc-1)-err(epoc));
                rate(epoc-1) = eta;
                diff_rate = abs(err(epoc) - err(1))/err(1);
                if DEBUG; fprintf('Iter %d, f(x) = %f, error_var = %f\n', i-1, err(epoc), evar(epoc-1)); end
                % check stopping condition
                if diff_rate > ratio
                    break; 
                end                        
            end
        end
    end
    fprintf('Iter %d\n', i-1);
    eps_0 = err(1);
    eps_e = err(epoc);    
    % update weights of training pairs
    alpha(n) = log((1-eps_e)/(eps_e+eps));
    Sc = Hx'*Hy;
    A = A.*exp(alpha(n)*(abs(Sc-S).^1));
    A = A*N*N/sum(A(:));
    tElapsed = toc(tStart);
    totalTime = totalTime + tElapsed;
    fprintf('%d iterations, %.5f seconds, error %.4f->%.4f\n', i-1, tElapsed, eps_0, eps_e);
    Px{n} = Wx;    
    Py{n} = Wy;    
end
model.Wx = cell2mat(Px);
model.Wy = cell2mat(Py);

avTime = totalTime / L;
fprintf('Training complete. Average time: %.5f seconds per bit.\n', avTime);
