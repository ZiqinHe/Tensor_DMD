%% tdmd_all_methods_err_table.m
% FFT-TDMD + Haar-TDMD + Cosine-TDMD + Unfold(bcirc whole) + TT-DMD
% Output:
%   1) per-frame relative error table for all frames t=1..r
%   2) 2x5 plot at a selected frame:
%      Row 1: reconstruction
%      Row 2: absolute error

clear; clc; close all;

%% ---------------- Load data ----------------
S = load('sst_60x60xT_sub.mat');
X = S.X;                               % n x m x T
[n,m,T] = size(X);

X_t = X(:,:,1:T-1);                    % n x m x r
Y_t = X(:,:,2:T);                      % n x m x r
r   = T-1;
a   = min(n,r);

% t-product view: n x r x m  (3rd dim = m)
X_sw = permute(X_t,[1 3 2]);           % n x r x m
Y_sw = permute(Y_t,[1 3 2]);           % n x r x m

realize = @(A) real(A);
relerr  = @(A,B) norm(A(:)-B(:),'fro') / max(norm(B(:),'fro'), eps);

%% =====================================================================
% 1) FFT-TDMD (slice-wise in Fourier domain)
% =====================================================================
fX = fft(X_sw, [], 3);
fY = fft(Y_sw, [], 3);

Phi_fft = zeros(n,a,m);
Lam_fft = zeros(a,a,m);
b_fft   = zeros(a,1,m);

for i = 1:m
    [u,s,v] = svd(fX(:,:,i),'econ');
    Atil    = u' * fY(:,:,i) * v * pinv(s);
    [W,D]   = eig(Atil);
    Phi_fft(:,:,i) = u * W;
    Lam_fft(:,:,i) = D;
end

X0_fft = fft(X_sw(:,1,:), [], 3);      % n x 1 x m
for i = 1:m
    b_fft(:,1,i) = pinv(Phi_fft(:,:,i)) * X0_fft(:,1,i);
end

% reconstruct all frames t=1..r in freq slices then ifft
Xt_fft = zeros(n,r,m);
for t = 1:r
    for i = 1:m
        Xt_fft(:,t,i) = Phi_fft(:,:,i) * (Lam_fft(:,:,i)^t) * b_fft(:,:,i);
    end
end
Xrec_fft_all = ifft(Xt_fft, [], 3);    % n x r x m

%% =====================================================================
% 2) Haar-TDMD (slice-wise after Haar block transform)
% =====================================================================
if mod(m,2) ~= 0
    error('Haar needs 3rd dim even, but m=%d', m);
end

Xh = fast_bdwt_haar_blocks(X_sw);
Yh = fast_bdwt_haar_blocks(Y_sw);

Phi_h = zeros(n,a,m);
Lam_h = zeros(a,a,m);
b_h   = zeros(a,1,m);

for i = 1:m
    [u,s,v] = svd(Xh(:,:,i),'econ');
    Atil    = u' * Yh(:,:,i) * v * pinv(s);
    [W,D]   = eig(Atil);
    Phi_h(:,:,i) = u * W;
    Lam_h(:,:,i) = D;
end

X0_h = fast_bdwt_haar_blocks(X_sw(:,1,:));
for i = 1:m
    b_h(:,1,i) = pinv(Phi_h(:,:,i)) * X0_h(:,1,i);
end

Xt_h = zeros(n,r,m);
for t = 1:r
    for i = 1:m
        Xt_h(:,t,i) = Phi_h(:,:,i) * (Lam_h(:,:,i)^t) * b_h(:,:,i);
    end
end
Xrec_haar_all = fast_bdwt_haar_blocks_inv(Xt_h);  % n x r x m

%% =====================================================================
% 3) Cosine-TDMD (slice-wise after cosine transform)
% =====================================================================
Xc = cosine_transform(X_sw);
Yc = cosine_transform(Y_sw);

Phi_c = zeros(n,a,m);
Lam_c = zeros(a,a,m);
b_c   = zeros(a,1,m);

for i = 1:m
    [u,s,v] = svd(Xc(:,:,i),'econ');
    Atil    = u' * Yc(:,:,i) * v * pinv(s);
    [W,D]   = eig(Atil);
    Phi_c(:,:,i) = u * W;
    Lam_c(:,:,i) = D;
end

X0_c = cosine_transform(X_sw(:,1,:));
for i = 1:m
    b_c(:,1,i) = pinv(Phi_c(:,:,i)) * X0_c(:,1,i);
end

Xt_c = zeros(n,r,m);
for t = 1:r
    for i = 1:m
        Xt_c(:,t,i) = Phi_c(:,:,i) * (Lam_c(:,:,i)^t) * b_c(:,:,i);
    end
end
Xrec_cos_all = inverse_cosine_transform(Xt_c);    % n x r x m

%% =====================================================================
% 4) Unfolding DMD via WHOLE bcirc matrix
%    use full (nm) x (r*m)
% =====================================================================
Xb = bcirc(X_sw);      % (n*m) x (r*m)
Yb = bcirc(Y_sw);      % (n*m) x (r*m)
rm = r*m;

[Uu,Su,Vu] = svd(Xb, 'econ');
Atil_unf   = Uu' * Yb * Vu * pinv(Su);
[W_unf,D_unf] = eig(Atil_unf);
Phi_unf    = Uu * W_unf;              % (nm) x (rm)
b_unf      = pinv(Phi_unf) * Xb(:,1); % initial = first col of bcirc

% store unfolding recon frames: n x m x r
Xrec_unf_frames = zeros(n,m,r);
for t = 1:r
    x_t = Phi_unf * (D_unf^t) * b_unf;            % (nm) x 1
    Xrec_unf_frames(:,:,t) = reshape(realize(x_t), [n,m]);
end

%% =====================================================================
% 5) TT-DMD
%    use exact block: Xmat = Xb(:,1:r), Ymat = Yb(:,1:r)
% =====================================================================
Xmat = Xb(:, 1:r);
Ymat = Yb(:, 1:r);

eps_tt = 1e-31;
Xtt = tt_tensor(X_t, eps_tt);
G   = core2cell(Xtt);
for k = 1:numel(G)
    G{k} = ensure_core3(G{k});
end
G = tt_left_orth_cells(G);

U = tt_contract_left_to_matrix(G(1:2), [n,m]);  % (nm) x rk
G3 = ensure_core3(G{3});
Xtime = squeeze(G3);                             % rk x r

[Us,S_tt,Vs] = svd(Xtime, 'econ');
Ueff = U * Us;
Veff = Vs;

Atil_ttd = (Ueff') * (Ymat * Veff) / S_tt;
[W_ttd,D_ttd] = eig(Atil_ttd);
Phi_ttd = (Ymat * Veff) / S_tt * W_ttd;          % (nm) x k
b_ttd   = pinv(Phi_ttd) * Xmat(:,1);

Xrec_ttd_frames = zeros(n,m,r);
for t = 1:r
    x_t = Phi_ttd * (D_ttd^t) * b_ttd;            % (nm) x 1
    Xrec_ttd_frames(:,:,t) = reshape(realize(x_t), [n,m]);
end

%% =====================================================================
% Error table for all frames t=1..r (compare to Y_t(:,:,t))
% =====================================================================
E_fft  = zeros(r,1);
E_haar = zeros(r,1);
E_cos  = zeros(r,1);
E_unf  = zeros(r,1);
E_ttd  = zeros(r,1);

for t = 1:r
    Ytrue = Y_t(:,:,t);

    % FFT/Haar/Cos are n x r x m -> take frame t then reshape to n x m
    fft_t  = realize( reshape(Xrec_fft_all(:,t,:),  [n,m]) );
    haar_t = realize( reshape(Xrec_haar_all(:,t,:), [n,m]) );
    cos_t  = realize( reshape(Xrec_cos_all(:,t,:),  [n,m]) );

    unf_t  = Xrec_unf_frames(:,:,t);
    ttd_t  = Xrec_ttd_frames(:,:,t);

    E_fft(t)  = relerr(fft_t,  Ytrue);
    E_haar(t) = relerr(haar_t, Ytrue);
    E_cos(t)  = relerr(cos_t,  Ytrue);
    E_unf(t)  = relerr(unf_t,  Ytrue);
    E_ttd(t)  = relerr(ttd_t,  Ytrue);
end

% MATLAB table
ErrTable = table((1:r)', E_fft, E_haar, E_cos, E_unf, E_ttd, ...
    'VariableNames', {'t','FFT_TDMD','Haar_TDMD','Cos_TDMD','Unfold_bcirc','TTD_TT_DMD'});

disp('================ Relative Error per frame (t=1..r) ================');
disp(ErrTable);

disp('---------------- Mean / Max over frames ----------------');
MeanRow = [mean(E_fft) mean(E_haar) mean(E_cos) mean(E_unf) mean(E_ttd)];
MaxRow  = [max(E_fft)  max(E_haar)  max(E_cos)  max(E_unf)  max(E_ttd)];
Summary = array2table([MeanRow; MaxRow], ...
    'VariableNames', {'FFT_TDMD','Haar_TDMD','Cos_TDMD','Unfold_bcirc','TTD_TT_DMD'}, ...
    'RowNames', {'Mean','Max'});
disp(Summary);

%% =====================================================================
% Plot: 2x5 at a specific t (e.g., t=6)
%   Row1: recon (FFT, Haar, Cos, UNF, TTD)
%   Row2: abs error |Recon - True|
%   True is Y_t(:,:,t)
% =====================================================================
tt = 6;
tt = min(tt, r);
Ytrue = realize(Y_t(:,:,tt));

% --- extract each method's reconstruction at frame tt ---
fft_t  = realize( reshape(Xrec_fft_all(:,tt,:),  [n,m]) );
haar_t = realize( reshape(Xrec_haar_all(:,tt,:), [n,m]) );
cos_t  = realize( reshape(Xrec_cos_all(:,tt,:),  [n,m]) );

unf_t  = realize( Xrec_unf_frames(:,:,tt) );
ttd_t  = realize( Xrec_ttd_frames(:,:,tt) );

R = {fft_t, haar_t, cos_t, unf_t, ttd_t};
E = {abs(fft_t-Ytrue), abs(haar_t-Ytrue), abs(cos_t-Ytrue), abs(unf_t-Ytrue), abs(ttd_t-Ytrue)};
names = {'TDMD(T)','TDMD(W)','TDMD(C)','DMD','TTD-DMD'};

% --- shared color limits ---
all_rec = [];
all_err = [];
for j = 1:5
    all_rec = [all_rec; R{j}(:)]; %#ok<AGROW>
    all_err = [all_err; E{j}(:)]; %#ok<AGROW>
end
cl_rec = local_clim_safe(all_rec);
cl_err = local_clim_safe(all_err);

% ------------------------ FIGURE SETTINGS ------------------------
fsTitle = 18;
fsErrTitle = 16; %#ok<NASGU>
fsCB = 14;

figure('Color','w','Position',[120 120 1650 520]);
colormap(parula(256));

tl = tiledlayout(2,5,'Padding','compact','TileSpacing','compact'); %#ok<NASGU>

% Row 1: recon
for j = 1:5
    nexttile(j);
    imagesc(R{j}, cl_rec);
    axis image off;
    title(names{j}, 'Interpreter','none', 'FontSize', fsTitle);
end

% Row 2: abs error
for j = 1:5
    nexttile(5+j);
    imagesc(E{j}, cl_err);
    axis image off;
    % title('|Recon - True|', 'Interpreter','none', 'FontSize', fsErrTitle);
end

cb = colorbar;
cb.Layout.Tile = 'east';
cb.FontSize = fsCB;
set(gcf, 'Color', 'w');

%% =====================================================================
% LOCAL FUNCTIONS
% =====================================================================

function C = ensure_core3(C)
    if ndims(C) == 2
        C = reshape(C, size(C,1), size(C,2), 1);
    end
end

function Gout = tt_left_orth_cells(Gin)
    d = numel(Gin);
    Gout = Gin;
    for k = 1:d-1
        Gk = ensure_core3(Gout{k});
        [rk,nk,rkp1] = size(Gk);

        M = reshape(Gk, rk*nk, rkp1);
        [Q,Rk] = qr(M,0);
        rnew = size(Q,2);
        Gout{k} = reshape(Q, rk, nk, rnew);

        Gnext = ensure_core3(Gout{k+1});
        [rkp1_check,nk2,rkp2] = size(Gnext);
        if rkp1_check ~= rkp1
            error('TT core mismatch at k=%d: right-rank=%d but next left-rank=%d.', ...
                  k, rkp1, rkp1_check);
        end
        Gnext_mat = reshape(Gnext, rkp1, []);
        Gnext_mat = Rk * Gnext_mat;
        Gout{k+1} = reshape(Gnext_mat, rnew, nk2, rkp2);
    end
end

function U = tt_contract_left_to_matrix(Gleft, nleft)
    G1 = ensure_core3(Gleft{1});
    U = reshape(G1, nleft(1), size(G1,3));
    for k = 2:numel(Gleft)
        Gk = ensure_core3(Gleft{k});
        [rk,nk,rkp1] = size(Gk);
        Gk2 = reshape(Gk, rk, nk*rkp1);
        U = U * Gk2;
        U = reshape(U, [], rkp1);
    end
end

function clim = local_clim_safe(A)
    A = double(A(:));
    A = A(isfinite(A));
    if isempty(A)
        clim = [0 1];
        return;
    end
    q = quantile(A,[0.02 0.98]);
    if ~isfinite(q(1)) || ~isfinite(q(2)) || q(2) <= q(1)
        clim = [min(A), max(A)];
        if clim(2) <= clim(1)
            clim = [0 1];
        end
    else
        clim = q;
    end
end