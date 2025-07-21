%%
clear
clc
load('video_data.mat'); 
V = X(1:60,1:60,1:20);
[n,m,T] = size(V);
X = V(:,:,1:T-1);
Y = V(:,:,2:T);
X_switched = permute(X, [1 3 2]);
Y_switched = permute(Y, [1 3 2]);

fX = fft(X_switched,[],3);
fY = fft(Y_switched,[],3);
a = min(n,T-1);
fA = zeros(a,a,m);
%for i = 1:m
%    [u,s,v] = svd(fX(:,:,i),'econ');
%    fA(:,:,i) = u' * fY(:,:,i)*v*inv(s);
%end
%A1 = ifft(fA,[],3);

Phi_fft = zeros(n,a,m);     % Tensor DMD modes in Fourier domain
Lambda_fft = zeros(size(fA));  % Diagonal eigenvalue matrix

for i = 1:m
    [u,s,v] = svd(fX(:,:,i),'econ');
    fA(:,:,i) = u' * fY(:,:,i)*v*pinv(s);
    [W,D] = eig(fA(:,:,i));
    Phi_fft(:,:,i) = u * W;
    Lambda_fft(:,:,i) = D;
end

% Inverse FFT to get Phi and Lambda in time domain
Phi = ifft(Phi_fft, [], 3);
Lambda = ifft(Lambda_fft, [], 3);

% FFT of initial state
X0_fft = fft(X_switched(:,1,:), [], 3);

% Solve Phi * b = X0 for b in Fourier domain
b_fft = zeros(a, 1, m);
for i = 1:m
    b_fft(:,1,i) = pinv(Phi_fft(:,:,i)) * X0_fft(:,1,i);  % b = Phi^{-1} * x0
end

% Convert back to time domain if needed
%b = ifft(b_fft, [], 3);


% Reconstruct X for each time step
Xt_fft = zeros(n,T-1,m);
X_dmd = zeros(n,T-1,m);
for t = 1:T-1
    for i = 1:m
        Xt_fft(:,t,i) = Phi_fft(:,:,i) * Lambda_fft(:,:,i)^t* b_fft(:,:,i);
    end    
    X_dmd(:,t,:) = ifft(Xt_fft(:,t,:), [], 3);
end
norm(bcirc(X_dmd - Y_switched), 'fro') / norm(bcirc(Y_switched), 'fro')

%%
frame_errors = zeros(T-1, 1);
for t = 1:T-1
    xt_dmd = reshape(X_dmd(:, t, :), [n, m]);
    yt_true = reshape(Y_switched(:, t, :), [n, m]);
    frame_errors(t) = norm(xt_dmd - yt_true, 'fro') / norm(yt_true, 'fro');
end

figure;
plot(1:T-1, frame_errors, 'o-','LineWidth',1.5);
xlabel('Frame Index t');
ylabel('Relative Reconstruction Error');
title('Per-frame DMD Reconstruction Error');
grid on;

%% unfolding based method
Xb = bcirc(X_switched);
Yb = bcirc(Y_switched);
% Step 1: SVD of X
[Ub, Sb, Vb] = svd(Xb, 'econ');

% Step 2: Project A onto reduced subspace
Ab_tilde = Ub' * Yb * Vb / Sb;

% Step 3: Eigendecomposition
[Wb, Db] = eig(Ab_tilde);         % D contains eigenvalues
Phib = Ub * Wb;                   % DMD modes (in full space)

% Step 4: Initial condition in block circulant form
x1b = bcirc(X_switched(:,1,:));  % Initial state reshaped into block circulant form

% Step 5: Compute mode amplitudes (b) in reduced coordinates
b_b = pinv(Phib) * x1b;

% Step 6: Time evolution in the unfolded space
Xb_dmd = zeros(size(Yb));  % Reconstructed Y
for t = 1:T-1
    Xb_dmd(:, (t-1)*m+1 : t*m) = Phib * (Db^t * b_b);  % Block-by-block
end

rel_error_unfold = norm(Xb_dmd - Yb, 'fro') / norm(Yb, 'fro');
fprintf('Relative error (Unfolding-based DMD): %.4e\n', rel_error_unfold);

%%
frame_errors = zeros(T-1, 1);
for t = 1:T-1
    xt_dmd = reshape(X_dmd(:, t, :), [n, m]);
    yt_true = reshape(Y_switched(:, t, :), [n, m]);
    frame_errors(t) = norm(xt_dmd - yt_true, 'fro') / norm(yt_true, 'fro');
end

frame_errors_unfold = zeros(T-1, 1);
for t = 1:T-1
    % Reconstructed and true blocks for frame t
    xtb_dmd = Xb_dmd(:, (t-1)*m+1 : t*m);
    ytb_true = Yb(:, (t-1)*m+1 : t*m);
    
    % Relative Frobenius norm error
    frame_errors_unfold(t) = norm(xtb_dmd - ytb_true, 'fro') / norm(ytb_true, 'fro');
end

% Plot frame-wise errorfigure;

plot(1:T-1, frame_errors, 'o-', 'LineWidth', 1.5); hold on;
plot(1:T-1, frame_errors_unfold, 's-', 'LineWidth', 1.5);
xlabel('Frame Index t');
ylabel('Relative Reconstruction Error');
title('Per-frame DMD Reconstruction Error Comparison');
legend('Tensor DMD', 'Unfolding DMD', 'Location', 'northwest');
grid on;
set(gcf, 'Color', 'w');  % Optional: white background for export

%% bg and fg
% Threshold for identifying static (background) modes
lambda_thresh = 0.995;

is_bg = false(size(Lambda_fft));  % same size as Lambda_fft

for i = 1:m
    lambda_abs = abs(diag(Lambda_fft(:,:,i)));
    is_bg(1:length(lambda_abs),1,i) = lambda_abs > lambda_thresh;
end


Xt_bg_fft = zeros(n, T-1, m);
for t = 1:T-1
    for i = 1:m
        Phi_i = Phi_fft(:,:,i);
        Lambda_i = Lambda_fft(:,:,i);
        b_i = b_fft(:,:,i);

        % Keep only background modes
        % Extract eigenvalues and zero out non-background ones
        lambda = diag(Lambda_i);              % vector of eigenvalues
        lambda(~is_bg(:,1,i)) = 0;            % zero out non-background modes
        Lambda_bg = diag(lambda);             % make it diagonal matrix

        Xt_bg_fft(:,t,i) = Phi_i * (Lambda_bg^t) * b_i;

    end
end

% Inverse FFT to get background in time domain
X_bg = real(ifft(Xt_bg_fft, [], 3));  % n × (T−1) × m


% X_bg: n × T−1 × m → permute back to original shape: n × m × T−1
V_bg = permute(X_bg, [1 3 2]);

% Pad background with 1 frame to match full video size
%V_bg_full = cat(3, V_bg, V_bg(:,:,end));  % n × m × T

% Foreground = original - background
V_fg = real(V(:,:,1:T-1) - V_bg);

%%
% Specify frame indices to visualize
frames_to_show = [1,10];

figure;
set(gcf, 'Color', 'w');
for i = 1:2
    t = frames_to_show(i);
    
    % Background subplot
    subplot(2,2,(i-1)*2+1);
    imagesc(V_bg(:,:,t));
    colormap(gray);
    axis image off;
    title(['Background Frame ', num2str(t)]);

    % Foreground subplot
    subplot(2,2,(i-1)*2+2);
    imagesc(V_fg(:,:,t));
    colormap(gray);
    axis image off;
    title(['Foreground Frame ', num2str(t)]);
end

sgtitle('Background and Foreground Separation');

%% Background/Foreground Separation in Unfolding DMD
lambda_thresh = 0.995;

% Identify background eigenvalues
lambda_vals = diag(Db);
is_bg = abs(lambda_vals) > lambda_thresh;

% Construct background evolution
Db_bg = diag(lambda_vals .* is_bg);  % Keep only background eigenvalues

% Reconstruct background
Xb_bg = zeros(size(Yb));
for t = 1:T-1
    Xb_bg(:, (t-1)*m+1 : t*m) = Phib * (Db_bg^t * b_b);
end

% Reconstruct background in video tensor form
X_bg_unfold = zeros(n, m, T-1);
%for t = 1:T-1
%    x_col = Xb_bg(:, (t-1)*m + 1);  % Each frame is in first column of each block
%    X_bg_unfold(:,:,t) = reshape(x_col, [n, m]);
%end
for t = 1:T-1
    X_bg_unfold(:,:,t) = Xb_bg((t-1)*n+1:t*n,1:m);
end
% Pad one frame to match original video size
%V_bg_unfold = cat(3, X_bg_unfold, X_bg_unfold(:,:,end));  % size: n × m × T

% Foreground = original - background
V_fg_unfold = real(V - V_bg_unfold);

%% Plot Frame 1 and 10 (Background + Foreground)
frames_to_show = [1, 10];

figure;
set(gcf, 'Color', 'w');

for i = 1:2
    t = frames_to_show(i);

    % Background subplot
    subplot(2, 2, (i-1)*2 + 1);
    imagesc(real(V_bg_unfold(:,:,t)));
    colormap(gray);
    axis image off;
    title(['Background Frame ', num2str(t)]);

    % Foreground subplot
    subplot(2, 2, (i-1)*2 + 2);
    imagesc(real(V_fg_unfold(:,:,t)));
    colormap(gray);
    axis image off;
    title(['Foreground Frame ', num2str(t)]);
end


sgtitle('Background and Foreground Separation (Unfolding DMD)');
