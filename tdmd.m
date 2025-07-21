
%% Averaged T-DMD and Unfolding DMD Performance over Multiple Trials
clear;
clc;
rng(1);  % For reproducibility

% Parameters
n = 25;
m = 10;
T = 20;
r = T - 1;
a = min(n,r);
p_list = 0:10;
num_p = length(p_list);
num_trials = 10;

avg_err_tdmd = zeros(1, num_p);
avg_time_tdmd = zeros(1, num_p);
avg_mem_tdmd = zeros(1, num_p);

avg_err_unfold = zeros(1, num_p);
avg_time_unfold = zeros(1, num_p);
avg_mem_unfold = zeros(1, num_p);

for trial = 1:num_trials
    % Generate synthetic tensor data
    Af = randn(n,n,m)*0.1;
    A = ifft(Af, [], 3);
    V = zeros(n,T,m);
    V(:,1,:) = randn(n,m);
    for i = 2:T
        V(:,i,:) = tprod(A, V(:,i-1,:));
    end
    X = V(:,1:T-1,:);
    Y = V(:,2:T,:);
    fX = fft(X,[],3);
    fY = fft(Y,[],3);
    Xb = bcirc(X);
    Yb = bcirc(Y);

    for pi = 1:num_p
        p = p_list(pi);

        %% Tensor DMD
        tic;
        fA = zeros(a,a,m);
        Phi_fft = zeros(n,a,m);
        Lambda_fft = zeros(a,a,m);
        mem_tdmd = 0;

        for i = 1:m
            [u,s,v] = svd(fX(:,:,i),'econ');
            k = size(s,1) - p;
            if k < 1, continue; end
            u_trunc = u(:,1:k);
            s_trunc = s(1:k,1:k);
            v_trunc = v(:,1:k);
            fA_temp = u_trunc' * fY(:,:,i) * v_trunc / s_trunc;
            [W,D] = eig(fA_temp);
            Phi_fft(:,1:k,i) = u_trunc * W;
            Lambda_fft(1:k,1:k,i) = D;
            mem_tdmd = mem_tdmd + numel(u_trunc)*k + numel(D);
        end

        X0_fft = fft(X(:,1,:), [], 3);
        b_fft = zeros(a,1,m);
        for i = 1:m
            b_fft(:,1,i) = pinv(Phi_fft(:,:,i)) * X0_fft(:,1,i);
        end

        Xt_fft = zeros(n,T-1,m);
        X_dmd = zeros(n,T-1,m);
        for t = 1:T-1
            for i = 1:m
                Xt_fft(:,t,i) = Phi_fft(:,:,i) * (Lambda_fft(:,:,i)^t) * b_fft(:,:,i);
            end
            X_dmd(:,t,:) = ifft(Xt_fft(:,t,:), [], 3);
        end
        rel_err_tdmd = norm(bcirc(X_dmd) - bcirc(Y), 'fro') / norm(bcirc(Y), 'fro');
        time_tdmd = toc;
        avg_err_tdmd(pi) = avg_err_tdmd(pi) + rel_err_tdmd;
        avg_time_tdmd(pi) = avg_time_tdmd(pi) + time_tdmd;
        avg_mem_tdmd(pi) = avg_mem_tdmd(pi) + mem_tdmd * 8 / (1024^2);  % MB

        %% Unfolding DMD
        tic;
        [Ub, Sb, Vb] = svd(Xb, 'econ');
        s = diag(Sb);
        k = max(length(s) - p*m, 1);
        Ub_trunc = Ub(:,1:k);
        Sb_trunc = Sb(1:k,1:k);
        Vb_trunc = Vb(:,1:k);
        Ab_tilde = Ub_trunc' * Yb * Vb_trunc / Sb_trunc;
        [Wb, Db] = eig(Ab_tilde);
        Phib = Ub_trunc * Wb;
        x1b = bcirc(X(:,1,:));
        b_b = pinv(Phib) * x1b;

        Xb_dmd = zeros(size(Yb));
        for t = 1:T-1
            Xb_dmd(:, (t-1)*m+1 : t*m) = Phib * (Db^t * b_b);
        end
        rel_err_unfold = norm(Xb_dmd - Yb, 'fro') / norm(Yb, 'fro');
        time_unfold = toc;
        mem_unfold = numel(Ub_trunc)*k + numel(Db);
        avg_err_unfold(pi) = avg_err_unfold(pi) + rel_err_unfold;
        avg_time_unfold(pi) = avg_time_unfold(pi) + time_unfold;
        avg_mem_unfold(pi) = avg_mem_unfold(pi) + mem_unfold * 8 / (1024^2);  % MB
    end
end

%% Average over trials
avg_err_tdmd = avg_err_tdmd / num_trials;
avg_time_tdmd = avg_time_tdmd / num_trials;
avg_mem_tdmd = avg_mem_tdmd / num_trials;

avg_err_unfold = avg_err_unfold / num_trials;
avg_time_unfold = avg_time_unfold / num_trials;
avg_mem_unfold = avg_mem_unfold / num_trials;

disp('Tensor DMD:')
disp(table(p_list', avg_err_tdmd', avg_time_tdmd', avg_mem_tdmd', ...
    'VariableNames', {'p', 'RelError', 'Time', 'Memory_MB'}));

disp('Unfolding DMD:')
disp(table(p_list', avg_err_unfold', avg_time_unfold', avg_mem_unfold', ...
    'VariableNames', {'p', 'RelError', 'Time', 'Memory_MB'}));



%%
% Tensor DMD with no truncation
fA = zeros(a, a, m);
Phi_fft = zeros(n, a, m);
Lambda_fft = zeros(a, a, m);

for i = 1:m
    [u, s, v] = svd(fX(:,:,i), 'econ');
    k = size(s, 1);  % keep all singular tubes
    u_trunc = u(:, 1:k);
    s_trunc = s(1:k, 1:k);
    v_trunc = v(:, 1:k);
    fA_temp = u_trunc' * fY(:,:,i) * v_trunc / s_trunc;
    [W, D] = eig(fA_temp);
    Phi_fft(:,1:k,i) = u_trunc * W;
    Lambda_fft(1:k,1:k,i) = D;
end

% Initial condition
X0_fft = fft(X(:,1,:), [], 3);
b_fft = zeros(a, 1, m);
for i = 1:m
    b_fft(:,1,i) = pinv(Phi_fft(:,:,i)) * X0_fft(:,1,i);
end

% Reconstruct
Xt_fft = zeros(n, T-1, m);
X_dmd = zeros(n, T-1, m);
for t = 1:T-1
    for i = 1:m
        Xt_fft(:,t,i) = Phi_fft(:,:,i) * (Lambda_fft(:,:,i)^t) * b_fft(:,:,i);
    end
    X_dmd(:,t,:) = ifft(Xt_fft(:,t,:), [], 3);
end

%% Plot first 4 Tensor DMD modes
n_modes_to_plot = 4;
figure;
set(gcf, 'Color', 'w');
for k = 1:n_modes_to_plot
    eigentuple_k = squeeze(Phi_fft(:,k,:));  % size: n x m
    subplot(1, n_modes_to_plot, k);
    imagesc(real(eigentuple_k));
    axis image off;
    title(['T-DMD Mode ', num2str(k)],'FontSize', 15);  % No bold
end


%% T-DMD vs Unfolding DMD: Per-frame Relative Error (No Truncation)
clear; clc;
rng(1);  % Reproducibility

% Parameters
n = 20; m = 10; T = 20; r = T - 1; a = min(n, T-1);


% Generate synthetic tensor data
Af = randn(n,n,m) * 0.1;
A = ifft(Af, [], 3);
V = zeros(n,T,m);
V(:,1,:) = randn(n,m);
for i = 2:T
    V(:,i,:) = tprod(A, V(:,i-1,:));  % tprod must be defined
end

% Prepare X and Y
X = V(:,1:T-1,:);
Y = V(:,2:T,:);

%% ---------- Tensor DMD ----------
X_fft = fft(X,[],3);
Y_fft = fft(Y,[],3);

Phi_fft = zeros(n, a, m);
Lambda_fft = zeros(a, a, m);
b_fft = zeros(a, 1, m);
for i = 1:m
    [u,s,v] = svd(X_fft(:,:,i), 'econ');
    Phi_fft(:,:,i) = u;
    Lambda_fft(:,:,i) = u' * Y_fft(:,:,i) * v / s;
end

% Initial condition in Fourier domain
X0_fft = fft(X(:,1,:),[],3);
for i = 1:m
    b_fft(:,1,i) = pinv(Phi_fft(:,:,i)) * X0_fft(:,1,i);
end

% Reconstruct each frame
Xt_fft = zeros(n,T-1,m);
X_dmd = zeros(n,T-1,m);
for t = 1:T-1
    for i = 1:m
        Xt_fft(:,t,i) = Phi_fft(:,:,i) * (Lambda_fft(:,:,i)^t) * b_fft(:,:,i);
    end
    X_dmd(:,t,:) = ifft(Xt_fft(:,t,:), [], 3);
end

% Compute frame-wise error
frame_errors_tdmd = zeros(T-1,1);
for t = 1:T-1
    x_dmd_t = reshape(X_dmd(:,t,:), [n,m]);
    y_true_t = reshape(Y(:,t,:), [n,m]);
    frame_errors_tdmd(t) = norm(x_dmd_t - y_true_t, 'fro') / norm(y_true_t, 'fro');
end

%% ---------- Unfolding DMD ----------
Xb = bcirc(X);
Yb = bcirc(Y);
[Ub, Sb, Vb] = svd(Xb, 'econ');
Ab_tilde = Ub' * Yb * Vb / Sb;
[Wb, Db] = eig(Ab_tilde);
Phib = Ub * Wb;

x1b = bcirc(X(:,1,:));
b_b = pinv(Phib) * x1b;

Xb_dmd = zeros(size(Yb));
for t = 1:T-1
    Xb_dmd(:,(t-1)*m+1:t*m) = Phib * (Db^(t+1) * b_b);
end
norm(Xb_dmd - Yb, 'fro') / norm(Yb, 'fro')
%%
% Frame-wise error for unfolding
frame_errors_unfold = zeros(T-1,1);
for t = 1:T-1
    xtb = Xb_dmd(:,(t-1)*m+1:t*m);
    ytb = Yb(:,(t-1)*m+1:t*m);
    frame_errors_unfold(t) = norm(xtb - ytb, 'fro') / norm(ytb, 'fro');
end

%% ---------- Plot ----------
figure;
plot(1:T-1, frame_errors_tdmd, 'o-', 'LineWidth', 1.5); hold on;
plot(1:T-1, frame_errors_unfold, 's-', 'LineWidth', 1.5);
xlabel('Frame Index t');
ylabel('Relative Reconstruction Error');
title('Per-frame Reconstruction Error: Tensor DMD vs Unfolding DMD');
legend('Tensor DMD', 'Unfolding DMD', 'Location', 'northeast');
grid on;
set(gcf, 'Color', 'w');
