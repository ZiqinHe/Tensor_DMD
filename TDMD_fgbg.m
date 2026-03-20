%% ========================================================================
%  FIVE-METHOD FG/BG @ frame 10 (2x5)
%  Methods: FFT-TDMD | Haar-TDMD | Cosine-TDMD | UNF(bcirc) | TTD(bcirc)
% ========================================================================

clear; clc; close all;
rng(1);

%% ---------------- Dimensions ----------------
n = 60; m = 60; T = 20; dt = 1;
r = T-1;
t_show = 10;
t_show = max(1, min(r, t_show));

%% ---------------- Synthetic BG (column-wise Fourier sparse) ----------------
bgBase = 0.25;
Kbg = [3 7 11 17];
ampK = 0.02;
row_phase = 2*pi*rand(n, numel(Kbg));
col_phase = 2*pi*rand(1, numel(Kbg));

%% ---------------- Foreground (moving square) ----------------
sq = 8;
row0 = 20;
col0 = 1;
col_speed = 1;
fgVal = 0.95;

%% ---------------- omega thresholds (BG = omega ~ 0) ----------------
tau_re = 2e-3;
tau_im = 0.02;

%% ---------------- TTD settings ----------------
hasTT = (exist('tt_matrix','file')==2) || (exist('tt_tensor','file')==2);
eps_tt = 1e-14;   % TT approx tolerance

%% 1) Build STATIC background BG (realistic common background)
%    - smooth illumination field (low-frequency)
%    - mild vignetting / gradient
%    - weak texture
%    - small sensor noise
bgBase = 0.25;

% ----- (A) Low-frequency illumination (smooth random field) -----
illum0 = randn(n,m);
h = fspecial('gaussian',[31 31], 7);          % big blur => low-freq
illum = imfilter(illum0, h, 'replicate');
illum = (illum - mean(illum(:))) / (std(illum(:)) + eps);

% ----- (B) Mild vignetting / global gradient (very common in cameras) -----
[xx,yy] = meshgrid(linspace(-1,1,m), linspace(-1,1,n));
rad = sqrt(xx.^2 + yy.^2);
vign = 1 - 0.25*rad.^2;                        % darker near corners

grad = 0.10*xx + 0.05*yy;                      % gentle tilt illumination

% ----- (C) Weak texture (band-limited noise) -----
tex0 = randn(n,m);
h2 = fspecial('gaussian',[9 9], 1.5);          % smaller blur => mid-freq
tex = imfilter(tex0, h2, 'replicate');
tex = (tex - mean(tex(:))) / (std(tex(:)) + eps);

% ----- Combine -----
BG = bgBase ...
   + 0.08*illum ...                             % smooth lighting variation
   + 0.04*tex ...                               % texture
   + 0.08*(vign - mean(vign(:))) ...            % vignetting
   + 0.05*grad;                                 % global gradient

% ----- (D) Small sensor noise (optional) -----
BG = BG + 0.01*randn(n,m);

% clamp to [0,1]
BG = max(0, min(1, BG));

%% 2) Build video V = BG + moving square
V = zeros(n,m,T);
for t = 1:T
    FG = zeros(n,m);
    c1 = col0 + (t-1)*col_speed;
    c1 = max(1, min(m - sq + 1, c1));
    c2 = c1 + sq - 1;

    r1 = max(1, min(n - sq + 1, row0));
    r2 = r1 + sq - 1;

    FG(r1:r2, c1:c2) = fgVal;
    V(:,:,t) = BG + FG;
end
V = max(0, min(1, V));

%% ------------------------ Visualize ORIGINAL frames ------------------------
figure('Color','w','Position',[100 100 1100 260]);
set(gcf, 'Color', 'w')

% frames to show
idxShow = [1, 2, T];   % T = 20

% layout parameters (journal-style)
ml   = 0.06;   % left margin
wImg = 0.24;   % image width
hImg = 0.70;   % image height
yImg = 0.15;   % vertical position
g    = 0.03;   % gap
wDot = 0.06;   % dots width

% x positions
x1 = ml;
x2 = x1 + wImg + g;
xDots = x2 + wImg + g;
x3 = xDots + wDot + g;

fsTitle = 14;
set(gcf, 'Color', 'w')

% -------- Frame 1 --------
ax1 = axes('Position',[x1 yImg wImg hImg]);
imagesc(V(:,:,idxShow(1)));
axis image off; colormap(gray);
title(sprintf('Frame %d', idxShow(1)), 'FontSize', fsTitle);

% -------- Frame 2 --------
ax2 = axes('Position',[x2 yImg wImg hImg]);
imagesc(V(:,:,idxShow(2)));
axis image off;
title(sprintf('Frame %d', idxShow(2)), 'FontSize', fsTitle);

% -------- dots (explicit three dots, journal-safe) --------
axDots = axes('Position',[xDots yImg+0.30 wDot 0.14]);
text(0.5, 0.5, '...', ...
     'FontSize', 28, ...
     'FontWeight','bold', ...
     'HorizontalAlignment','center', ...
     'VerticalAlignment','middle', ...
     'Color', [0.2 0.2 0.2]);
axis off;


% -------- Frame T --------
ax3 = axes('Position',[x3 yImg wImg hImg]);
imagesc(V(:,:,idxShow(3)));
axis image off;
title(sprintf('Frame %d', idxShow(3)), 'FontSize', fsTitle);

set([ax1,ax2,ax3],'PlotBoxAspectRatio',[1 1 1]);

%%
rowMean = mean(V, 2);                     % n x 1 x T
Vd  = V - repmat(rowMean, 1, m, 1);   % remove per-frame row DC

rowMean_bg = median(rowMean, 3);          % n x 1
BG_row = repmat(rowMean_bg, 1, m, T);     % n x m x T

X = Vd(:,:,1:T-1);
Y = Vd(:,:,2:T);

X_sw = permute(X, [1 3 2]);   % n x r x m
Y_sw = permute(Y, [1 3 2]);   % n x r x m

%% 5) TDMD variants: FFT / Cosine / Haar

% ---- FFT ----
fX_fft = fft(X_sw, [], 3);
fY_fft = fft(Y_sw, [], 3);
[V_bg_fft_detr, ~] = tdmd_fgbg_perbin(fX_fft, fY_fft, 'fft', n, m, T, dt, tau_re, tau_im);

% ---- Cosine (DCT) with padding to suppress reflection ghosts ----
mp = 2^nextpow2(2*m);                 % bigger than m (e.g. 128 when m=60)

X_pad = cat(3, X_sw, zeros(n, r, mp-m));
Y_pad = cat(3, Y_sw, zeros(n, r, mp-m));

fX_dct = dct(X_pad, [], 3);
fY_dct = dct(Y_pad, [], 3);

[V_bg_cos_detr_pad_fullT, ~] = tdmd_fgbg_perbin(fX_dct, fY_dct, 'cos', n, mp, T, dt, tau_re, tau_im);

% crop back to original width m
V_bg_cos_detr_fullT = V_bg_cos_detr_pad_fullT(:,1:m,:);   % n x m x T
V_bg_cos_detr = V_bg_cos_detr_fullT(:,:,1:T-1);

V_bg_cos = real(V_bg_cos_detr + BG_row(:,:,1:T-1));
V_fg_cos = max(real(V(:,:,1:T-1) - V_bg_cos), 0);


% addback + FG
V_bg_cos = real(V_bg_cos_detr + BG_row(:,:,1:T-1));
V_fg_cos = max(real(V(:,:,1:T-1) - V_bg_cos), 0);


% ---- Haar (kept your old way) ----
L = 2^nextpow2(m);        % 64
H = haar_orth_matrix(L);

X_pad = cat(3, X_sw, zeros(n, r, L-m));
Y_pad = cat(3, Y_sw, zeros(n, r, L-m));

fX_haar = mode3_matmul(X_pad, H');
fY_haar = mode3_matmul(Y_pad, H');

[V_bg_haar_detr_pad, ~] = tdmd_fgbg_perbin(fX_haar, fY_haar, 'haar', n, L, T, dt, tau_re, tau_im);

V_bg_haar_detr = V_bg_haar_detr_pad(:,1:m, :);

V_bg_fft = V_bg_fft_detr(:,:,1:T-1) + BG_row(:,:,1:T-1);
V_fg_fft = max(real(V(:,:,1:T-1) - V_bg_fft), 0);

V_bg_cos = V_bg_cos_detr(:,:,1:T-1) + BG_row(:,:,1:T-1);
V_fg_cos = max(real(V(:,:,1:T-1) - V_bg_cos), 0);

V_bg_haar = V_bg_haar_detr(:,:,1:T-1) + BG_row(:,:,1:T-1);
V_fg_haar = max(real(V(:,:,1:T-1) - V_bg_haar), 0);

%% 6) UNF(bcirc) DMD + BG/FG 
if exist('bcirc','file')~=2
    error('Need bcirc() on your path.');
end

Xb = bcirc(X_sw);
Yb = bcirc(Y_sw);

[Ub, Sb, Vb] = svd(Xb, 'econ');
Ab_tilde = Ub' * Yb * Vb * pinv(Sb);
[Wb, Db] = eig(Ab_tilde);
Phib = Ub * Wb;

x1b = bcirc(X_sw(:,1,:));
b_b = pinv(Phib) * x1b;

lam_vals = diag(Db);
omega_unf = log(lam_vals)/dt;
is_bg_unf = (abs(real(omega_unf)) < tau_re) & (abs(imag(omega_unf)) < tau_im);
if ~any(is_bg_unf)
    [~,id] = min(abs(omega_unf));
    is_bg_unf(id) = true;
end
Db_bg = diag(lam_vals .* is_bg_unf);

Xb_bg = zeros(size(Yb));
for t = 1:T-1
    Xb_bg(:, (t-1)*m+1 : t*m) = Phib * (Db_bg^t * b_b);
end

X_bg_unf_detr = zeros(n, m, T-1);
for t = 1:T-1
    X_bg_unf_detr(:,:,t) = Xb_bg((t-1)*n+1:t*n, 1:m);
end

V_bg_unf = real(X_bg_unf_detr + BG_row(:,:,1:T-1));
V_fg_unf = max(real(V(:,:,1:T-1) - V_bg_unf), 0);

%% 7) TTD
if ~hasTT
    warning('TT-Toolbox not found: skipping TTD.');
    V_bg_ttd = nan(n,m,T-1);
    V_fg_ttd = nan(n,m,T-1);
else
    Xb_tt = tt_matrix(Xb, eps_tt);
    Yb_tt = tt_matrix(Yb, eps_tt);
    Xb2 = full(Xb_tt);
    Yb2 = full(Yb_tt);

    [Ub2, Sb2, Vb2] = svd(Xb2, 'econ');
    Ab_tilde2 = Ub2' * Yb2 * Vb2 * pinv(Sb2);
    [Wb2, Db2] = eig(Ab_tilde2);
    Phib2 = Ub2 * Wb2;

    b_b2 = pinv(Phib2) * x1b;

    lam2 = diag(Db2);
    omega2 = log(lam2)/dt;
    is_bg2 = (abs(real(omega2)) < tau_re) & (abs(imag(omega2)) < tau_im);
    if ~any(is_bg2)
        [~,id] = min(abs(omega2));
        is_bg2(id) = true;
    end
    Db_bg2 = diag(lam2 .* is_bg2);

    Xb_bg2 = zeros(size(Yb2));
    for t = 1:T-1
        Xb_bg2(:, (t-1)*m+1 : t*m) = Phib2 * (Db_bg2^t * b_b2);
    end

    X_bg_ttd_detr = zeros(n,m,T-1);
    for t = 1:T-1
        X_bg_ttd_detr(:,:,t) = Xb_bg2((t-1)*n+1:t*n, 1:m);
    end

    V_bg_ttd = real(X_bg_ttd_detr + BG_row(:,:,1:T-1));
    V_fg_ttd = max(real(V(:,:,1:T-1) - V_bg_ttd), 0);
end

%% 8) PLOT: 2 rows x 5 cols at frame t_show
t = t_show;

FGs = { V_fg_fft(:,:,t),  V_fg_haar(:,:,t), V_fg_cos(:,:,t), V_fg_unf(:,:,t), V_fg_ttd(:,:,t) };
BGs = { V_bg_fft(:,:,t),  V_bg_haar(:,:,t), V_bg_cos(:,:,t), V_bg_unf(:,:,t), V_bg_ttd(:,:,t) };
names = {'TDMD(T)','TDMD(W)','TDMD(C)','DMD','TTD-DMD'};

figure('Color','w','Position',[100 100 1650 520]); colormap(gray);

for j = 1:5
    subplot(2,5,j);
    imagesc(real(FGs{j}) + 0.25); axis image off; set(gca,'CLim',[0 1]);
    title([names{j}, ' FG'], 'Interpreter','none');

    subplot(2,5,5+j);
    imagesc(real(BGs{j})); axis image off; set(gca,'CLim',[0 1]);
    title([names{j}, ' BG'], 'Interpreter','none');
end

%%                           LOCAL FUNCTIONS

function [V_bg_detr_fullT, V_fg_fullT] = tdmd_fgbg_perbin(fX, fY, which, n, m3, T, dt, tau_re, tau_im)
r = size(fX,2);
a = min(n, r);

Phi_cell = cell(1,m3);
lam_cell = cell(1,m3);
b_cell   = cell(1,m3);
bgidx_cell = cell(1,m3);

for i = 1:m3
    Xi = fX(:,:,i);
    Yi = fY(:,:,i);
    x1 = Xi(:,1);

    [U,S,Vv] = svd(Xi,'econ');
    k2 = min(a, size(S,1));
    U2 = U(:,1:k2);
    S2 = S(1:k2,1:k2);
    V2 = Vv(:,1:k2);

    Atil = U2' * Yi * V2 * pinv(S2);
    [W,D] = eig(Atil);

    Phi = U2 * W;
    lam = diag(D);
    b   = pinv(Phi) * x1;

    omega = log(lam)/dt;
    bg_idx = find(abs(real(omega)) < tau_re & abs(imag(omega)) < tau_im);
    if isempty(bg_idx)
        [~,id] = min(abs(omega));
        bg_idx = id;
    end

    Phi_cell{i} = Phi;
    lam_cell{i} = lam;
    b_cell{i}   = b;
    bgidx_cell{i} = bg_idx;
end

Zbg = zeros(n, T, m3);
for t = 1:T
    for i = 1:m3
        Phi = Phi_cell{i};
        lam = lam_cell{i};
        b   = b_cell{i};
        idx = bgidx_cell{i};
        Zbg(:,t,i) = Phi(:,idx) * ((lam(idx).^(t-1)) .* b(idx));
    end
end

switch lower(which)
    case 'fft'
        Xbg = real(ifft(Zbg, [], 3));              % n x T x m3
        V_bg_detr_fullT = permute(Xbg, [1 3 2]);   % n x m3 x T

    case 'cos'
        Xbg = real(idct(Zbg, [], 3));
         % Zbg is n x T x m3, cosine coeff along 3rd dim (m3)
        V_bg_detr_fullT = permute(Xbg, [1 3 2]);    % n x m3 x T


    case 'haar'
        % unchanged: returns coeff-domain; caller must inverse using H
        Xbg = Zbg;
        V_bg_detr_fullT = permute(Xbg, [1 3 2]);

    otherwise
        error('Unknown transform type.');
end

V_fg_fullT = zeros(size(V_bg_detr_fullT));
end

function Z = mode3_matmul(X, A)
[n,r,m] = size(X);
X2 = reshape(X, n*r, m);
Z2 = X2 * A;
Z  = reshape(Z2, n, r, m);
end

function H = haar_orth_matrix(L)
p = log2(L);
if abs(p - round(p)) > 1e-12
    error('Haar size must be power of 2.');
end
p = round(p);

H = zeros(L,L);
H(1,:) = 1/sqrt(L);

row = 2;
for j = 0:p-1
    block = 2^(p-j);
    half  = block/2;
    num   = 2^j;
    for k = 0:num-1
        idx = k*block + (1:block);
        H(row, idx(1:half)) =  1/sqrt(block/2);
        H(row, idx(half+1:end)) = -1/sqrt(block/2);
        row = row + 1;
    end
end
end

function V_rec_detr_fullT = tdmd_reconstruct_perbin(fX, fY, which, n, m3, T, dt)
% per-bin TDMD full reconstruction (use ALL modes, not only BG)
% Input fX,fY: n x r x m3  (coeff-domain snapshots, r=T-1)
% Output V_rec_detr_fullT: n x m3 x T (detrended, physical domain for fft/cos; coeff-domain for haar handled below)

r = size(fX,2);
a = min(n, r);

Phi_cell = cell(1,m3);
lam_cell = cell(1,m3);
b_cell   = cell(1,m3);

for i = 1:m3
    Xi = fX(:,:,i);
    Yi = fY(:,:,i);
    x1 = Xi(:,1);

    [U,S,Vv] = svd(Xi,'econ');
    k2 = min(a, size(S,1));
    U2 = U(:,1:k2);
    S2 = S(1:k2,1:k2);
    V2 = Vv(:,1:k2);

    Atil = U2' * Yi * V2 * pinv(S2);
    [W,D] = eig(Atil);

    Phi = U2 * W;
    lam = diag(D);
    b   = pinv(Phi) * x1;

    Phi_cell{i} = Phi;
    lam_cell{i} = lam;
    b_cell{i}   = b;
end

Zrec = zeros(n, T, m3);
for t = 1:T
    for i = 1:m3
        Phi = Phi_cell{i};
        lam = lam_cell{i};
        b   = b_cell{i};
        Zrec(:,t,i) = Phi * ((lam.^(t-1)) .* b);   % ALL modes
    end
end

switch lower(which)
    case 'fft'
        Xrec = real(ifft(Zrec, [], 3));              % n x T x m3
        V_rec_detr_fullT = permute(Xrec, [1 3 2]);   % n x m3 x T

    case 'cos'
        Xrec = real(idct(Zrec, [], 3));              % n x T x m3
        V_rec_detr_fullT = permute(Xrec, [1 3 2]);   % n x m3 x T

    case 'haar'
        % return coeff-domain here; caller must inverse with H
        V_rec_detr_fullT = permute(Zrec, [1 3 2]);   % n x m3 x T (still in Haar-coeff along 2nd dim)
    otherwise
        error('Unknown transform type.');
end
end


V_rec_fft_detr_fullT = tdmd_reconstruct_perbin(fX_fft, fY_fft, 'fft', n, m, T, dt);
V_rec_fft = real(V_rec_fft_detr_fullT(:,:,1:T-1) + BG_row(:,:,1:T-1));

V_rec_cos_detr_pad_fullT = tdmd_reconstruct_perbin(fX_dct, fY_dct, 'cos', n, mp, T, dt);
V_rec_cos_detr_fullT = V_rec_cos_detr_pad_fullT(:,1:m,:);      % crop
V_rec_cos = real(V_rec_cos_detr_fullT(:,:,1:T-1) + BG_row(:,:,1:T-1));

V_rec_haar_coeff_fullT = tdmd_reconstruct_perbin(fX_haar, fY_haar, 'haar', n, L, T, dt);
% inverse Haar along 2nd dim (the "m3" dim here)
V_rec_haar_detr_pad_fullT = mode3_matmul(permute(V_rec_haar_coeff_fullT, [1 3 2]), H); % n x T x L
V_rec_haar_detr_pad_fullT = permute(V_rec_haar_detr_pad_fullT, [1 3 2]);              % n x L x T
V_rec_haar_detr_fullT = V_rec_haar_detr_pad_fullT(:,1:m,:);                            % crop back
V_rec_haar = real(V_rec_haar_detr_fullT(:,:,1:T-1) + BG_row(:,:,1:T-1));

Db_all = diag(lam_vals);   % all modes

Xb_rec = zeros(size(Yb));
for t = 1:T-1
    Xb_rec(:, (t-1)*m+1 : t*m) = Phib * (Db_all^t * b_b);
end

X_rec_unf_detr = zeros(n, m, T-1);
for t = 1:T-1
    X_rec_unf_detr(:,:,t) = Xb_rec((t-1)*n+1:t*n, 1:m);
end

V_rec_unf = real(X_rec_unf_detr + BG_row(:,:,1:T-1));

Db_all2 = diag(lam2);   % all modes

Xb_rec2 = zeros(size(Yb2));
for t = 1:T-1
    Xb_rec2(:, (t-1)*m+1 : t*m) = Phib2 * (Db_all2^t * b_b2);
end

X_rec_ttd_detr = zeros(n,m,T-1);
for t = 1:T-1
    X_rec_ttd_detr(:,:,t) = Xb_rec2((t-1)*n+1:t*n, 1:m);
end

V_rec_ttd = real(X_rec_ttd_detr + BG_row(:,:,1:T-1));

%  Per-frame relative error (Frobenius) for reconstructions
errs_fft  = zeros(1,T-1);
errs_cos  = zeros(1,T-1);
errs_haar = zeros(1,T-1);
errs_unf  = zeros(1,T-1);
errs_ttd  = zeros(1,T-1);

for t = 1:T-1
    Vt = V(:,:,t);
    denom = norm(Vt,'fro') + eps;

    errs_fft(t)  = norm(V_rec_fft(:,:,t)  - Vt,'fro') / denom;
    errs_cos(t)  = norm(V_rec_cos(:,:,t)  - Vt,'fro') / denom;
    errs_haar(t) = norm(V_rec_haar(:,:,t) - Vt,'fro') / denom;
    errs_unf(t)  = norm(V_rec_unf(:,:,t)  - Vt,'fro') / denom;

    if exist('V_rec_ttd','var') && all(size(V_rec_ttd) == [n,m,T-1])
        errs_ttd(t) = norm(V_rec_ttd(:,:,t) - Vt,'fro') / denom;
    else
        errs_ttd(t) = nan;
    end
end

figure('Color','w','Position',[120 120 900 320]);
plot(1:T-1, errs_fft,  '-o', 'LineWidth',3); hold on;
plot(1:T-1, errs_haar, '-^', 'LineWidth',3);
plot(1:T-1, errs_cos,  '-d', 'LineWidth',3);
plot(1:T-1, errs_unf,  '-x', 'LineWidth',3);
plot(1:T-1, errs_ttd,  '--s', 'LineWidth',3);
grid on;
xlabel('Frame index t');
ylabel('Relative error  ||V_{rec}(:,:,t)-V(:,:,t)||_F / ||V(:,:,t)||_F');
legend({'FFT-TDMD','Haar-TDMD','Cosine-TDMD','UNF(bcirc)','TTD(bcirc)'}, ...
       'Location','northwest');
title('Per-frame reconstruction relative error (5 methods)');
