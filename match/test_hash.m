function [ap, mAP, pM, pr, re] = test_hash(D, S, L, opts_M)
%% compute performace metrics
% D: test training pairwise distance
% S: test training ground truth similarity label
% R: options for radius of hamming ball
% M: options for number of nearest neighbors 

[Ntest, Ndb] = size(D);
eps = 1e-10;
S = logical(S);

% average presion & preision-reall
prc = zeros(1, L); % preision
rec = zeros(1, L); % reall
for l = 1 : L
    Stmp = D <= l;
    prc(l) = sum(sum(S.*Stmp))/(eps+sum(Stmp(:)));
    rec(l) = sum(sum(S.*Stmp))/(eps+sum(S(:)));
end
ap = sum((prc(1:end-1)+prc(2:end))/2 .* (rec(2:end)-rec(1:end-1)));
% mAP and topM preision
[~, Id] = sort(D, 2);                
lid = sub2ind(size(D), repmat(1:Ntest, 1, Ndb), Id(:)');
Stmp = reshape(S(lid), Ntest, Ndb);
Scumsum = cumsum(Stmp, 2);
P =  Scumsum ./ repmat(1:Ndb, Ntest, 1);
P_50 = P(:, 1:50);
Stmp_50 = Stmp(:, 1:50);
mAP = mean(sum(P_50.*Stmp_50, 2) ./ (sum(Stmp_50, 2)+eps));
mp = mean(P);
pM = mp(opts_M);
% precision vs. recall
cum_rec = sum(Stmp, 1);
id_rec = find(cum_rec);
mR = sum(Scumsum, 1)/sum(Stmp(:));
pr = mp(id_rec);
re = mR(id_rec);