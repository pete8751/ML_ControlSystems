% Deep deterministic policy gradient, in a case where the discount factor
% gamma = 1.

% Calls create_net, forward, backprop, adam, d_dx, & batch_nse, & also
% the function test_policy at the bottom of this file.

clear variables;
clc;
clf;

global n_q psi zeta q_min q_max n_steps Dt r

% Set up state dynamics
n_q = 3; %number of components in dynamics.
n_s = 2*n_q; %Number of components of state vector (each component is a 2d vector).
n_a = 2; %Number of components of action vector. 
zeta = 0.75;
[psi_1, psi_2, psi_3] = deal(1.88, 0.6, 0.48);
psi = [psi_1, psi_2, psi_3];

q_min = [-2; 0]; % max and min joint constraints.
q_max = [1.5; 2.5];

% Define task
s_star = [0.5; 0.5; 0; 0; 0; 0];
r = @(s, a) -10 * (s - s_star)'*(s-s_star) - 0.01*a'*a;
% r = @(s, a) -s'*s - 0.001*a'*a;  % reward function
Dt = 0.1;
dur = 2;
n_steps = floor(1 + dur / Dt); %should I floor or ceil this? should I add 1?

% Set up learning
a_sd = 0.1;  % standard deviation of stochastic policy
eta_mu = 3e-8;  %1e-6; 
eta_f = 1e-4;
eta_r = 1e-2;
eta_V = 1e-4;
tau = 3e-4;
adam_1 = 0.9;
adam_2 = 0.999;
n_rollouts = 1000;
n_m = 100; %size of minibatches. 
n_buf = 100000; %MAX NUMBER OF EXAMPLES IN THE BUFFER. 
buf_fill = 0; %tracks number of examples in the buffer
buf_i = 0; %index for buffer columns. 
BUFFER = zeros(2*n_s + n_a + 1, n_buf); %[s_t;a_t;r(s_t,a_t);s_t+1]

% Set initial state for test
s_test = [-1; 1; 0; 0; 0; 0];

% Seed random-number generator
rng(4);  

% Create nets
rescale = 0.5*[0; 1; 1; 1];
mu = create_net([n_s; 250; 250; n_a], rescale, "relu"); %input is state, two hidden layers 100 width, output is action. This is the policy. 
f_est = create_net([n_s + n_a; 100; 100; n_s], rescale, "relu"); %input is current state and action, output is estimate of f(s_t, a_t)
r_est = create_net([n_s + n_a; 50; 50; 1], rescale, "relu"); %input is current state and action, output is estimate of r(s_t, a_t)
V_est = create_net([n_s; 100; 100; 1], rescale, "relu"); %input is state, output is estimate of V^(mu)(s_t)

f_tgt = f_est; %Target network for f
V_tgt = V_est; %Target network for V

% Set up graphic
figure(1);
set(gcf, 'Name', 'DDPG', 'NumberTitle', 'off');

% Assess initial policy
G = test_policy(mu, s_test); %executed before the first rollout, and then after every 100
fprintf('At rollout 0, G = %.3f\n', G);

% Train policy
for rollout = 1:n_rollouts

  % Run rollout

  %initialization with random position, and slight random vel/acc. 
   s = [q_min(1); q_min(2); -0.1; -0.1; -0.1; -0.1] +...
       ([q_max(1) - q_min(1); q_max(2) - q_min(2); 0.2; 0.2; 0.2; 0.2].*(rand(n_s, 1)));

  for t = 1:n_steps %Number of iterations 

    % Compute transition
    mu = forward(mu, s);
    a = mu.y{end} + a_sd*randn(n_a, 1); %finding predicted action using mu network, and then adding stochastic noise to all components scaled by a_d. 
    s_next = f(s, a); %s_next is computed by using actual values for policy and state. 

    % Store in the buffer
    BUFFER(:, buf_i + 1) = [s; a; r(s, a); s_next]; %puts current example column into the buffer.
    buf_i = mod(buf_i + 1, n_buf); %sets index of next insert, resetting to 0 after it exceeds buffer size, hence will replace oldest example next time. 
    buf_fill = min(buf_fill + 1, n_buf); %keeps track of how full the buffer is (how many examples are in the buffer).

    % Choose a minibatch from the buffer
    i = ceil(buf_fill*rand(1, n_m)); %creates an array of n_m numbers between 0 and 1, multiplies by buf capacity, takes ceiling producing indices for minibatch.
    s_ = BUFFER(1:n_s, i); %selects all rows from 1 to n_s, at columns given in array i, and these are the s_t examples. 
    a_ = BUFFER(n_s + 1:n_s + n_a, i); %selects all rows from the first component of a to the last at columns at array i, these are a_t examples. 
    r_ = BUFFER(n_s + n_a + 1:n_s + n_a + 1, i); %selects row corresponding to r value for all columns in array i, these are r(s_t,a_t) examples. 
    s_next_ = BUFFER(n_s + n_a + 2:end, i); %these are s_t+1 examples at columns in array i indices. 

    % Adjust reward network (i.e. r_est) to minimize the squared Bellman error over the buffer-minibatch
    r_est = forward(r_est, [s_; a_]); %run state and action minibatch through the Q_est network. 
    r_error = r_est.y{end} - r_;
    r_est = backprop(r_est, r_error); %Backpropagate this error through the r_est neural network.
    r_est = adam(r_est, r_est.dL_dW, r_est.dL_db, eta_r, adam_1, adam_2); %use adam to adjust parameters of the neural network. 

    % Adjust critic (i.e. V_est) to minimize the squared Bellman error over the buffer-minibatch
    V_est = forward(V_est, s_); %run state components of minibatch through the V_est network. 
    mu = forward(mu, s_); %run the s_t+1 examples through the mu policy network (batched).
    a_next_ = mu.y{end} + a_sd*randn(n_a, n_m);
    r_est = forward(r_est, [s_; a_next_]);
    policy_r = r_est.y{end};
    f_tgt = forward(f_tgt, [s_; a_next_]); %replace the second argument with variable?
    V_tgt = forward(V_tgt, f_tgt.y{end});
    V_error = V_est.y{end} - (policy_r*Dt) - V_tgt.y{end}; %
    V_est = backprop(V_est, V_error);
    V_est = adam(V_est, V_est.dL_dW, V_est.dL_db, eta_V, adam_1, adam_2); %use adam to adjust parameters of the neural network. 
    
    % Adjust next-state dynamic network (i.e. f_est) to improve next state
    f_est = forward(f_est, [s_; a_]); %run state and action minibatch through the Q_est network. 
    V_est = forward(V_est, f_est.y{end});
    f_pred = V_est.y{end};
    dV_df = d_dx(V_est, ones(1, n_m));
    V_est = forward(V_est, s_next_);
    f_star = V_est.y{end};
    f_error = f_pred - f_star;
    dferror_dV = dV_df .* f_error; %If you simply backprop Dv_df the model learns much better.
    f_est = backprop(f_est, dferror_dV); %Backpropagate this error through the f_est neural network.
    f_est = adam(f_est, f_est.dL_dW, f_est.dL_db, eta_f, adam_1, adam_2); %use adam to adjust parameters of the neural network. 

    % Adjust actor (i.e. policy, mu) to maximize Q over buffer-minibatch
    mu = forward(mu, s_); %run policy over given states 
    pol_ = mu.y{end}; %get batch of output states for policy
    f_est = forward(f_est, [s_; pol_]); %run state and action minibatch through the Q_est network. 
    V_est = forward(V_est, f_est.y{end});
    dV_df = d_dx(V_est, ones(1, n_m)); %Calculate delta_V/d_f[s_i;a_i] for all examples in the minibatch by backpropping 1s through the network.
    dV = d_dx(f_est, dV_df); %Calculate delta_f/d_[s_i;a_i] for all examples in the minibatch by backpropping 1s through the network.
    
    r_est = forward(r_est, [s_; pol_]);
    dr = d_dx(r_est, ones(1, n_m));

    dV_da = dV(n_s + 1:end, :);
    dr_da = dr(n_s + 1:end, :);
    dQ_da = dV_da + dr_da*Dt;

    mu = backprop(mu, -dQ_da); %Backprop these gradients through the policy network to get derivative of of policy w.r.t its parameters. WHY NEGATIVE???
    mu = adam(mu, mu.dL_dW, mu.dL_db, eta_mu, adam_1, adam_2); %adjust parameters of policy. 
               
    % Nudge target net toward learning one
    for l = 2:V_est.n_layers %move weights and biases of target networks towards estimate networks. (why???)
      V_tgt.W{l} = V_tgt.W{l} + tau*(V_est.W{l} - V_tgt.W{l});
      V_tgt.b{l} = V_tgt.b{l} + tau*(V_est.b{l} - V_tgt.b{l});
      f_tgt.W{l} = f_tgt.W{l} + tau*(f_est.W{l} - f_tgt.W{l});
      f_tgt.b{l} = f_tgt.b{l} + tau*(f_est.b{l} - f_tgt.b{l});
    end
        
    % Update s
    s = s_next;

  end  % for t
  
  % Test policy
  if mod(rollout, 100) == 0
    %have to print r_nse, f_nse and V_nse.
    r_nse = batch_nse(r_ , r_error);
    f_nse = batch_nse(f_star, f_error);
    V_nse = batch_nse(policy_r*Dt + V_tgt.y{end}, V_error);
    G = test_policy(mu, s_test);  
    fprintf('At rollout %d, G = %.3f\n, %.4f\n, %.4f\n, %.4f\n', rollout, G, r_nse, f_nse, V_nse)
  end

end  % for rollout

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function G = test_policy(mu, s)
  
  global n_steps Dt r

  DATA = zeros(5, n_steps);
  G = 0; %reset G at each call of test_policy to zero. 
  for t = 1:n_steps %iterates through n_steps iterations. 
    mu = forward(mu, s);  % set noise to 0 for test //run through the network to get a predicted best action.
    a = mu.y{end}; %this action is the output of the policy network given the current state. 
    reward = Dt * r(s, a);
    G = G + reward; %The gain is the current gain plus the reward of the action.
    DATA(:, t) = [t; s(1); s(2); a]; %WHY DOES IT ONLY TAKE THE FIRST ROW OF THE STATE VECTOR?
    s = f(s, a); %This calculates the actual next state based on current state, action and state dynamics. 
  end  % for t
      

end  % function test_policy

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function s_next = f(s, a)

  global psi zeta q_min q_max Dt

    %computes M Matrix on rhs of torque equation, (first and second derivative of q)
    M = @(psi_1, psi_2, psi_3, q)[psi_1 + (2 * psi_2 * cos(q(2))), psi_3 + (psi_2 * cos(q(2))); 
    psi_3 + (psi_2 * cos(q(2))), psi_3];

    delta_M = @(psi_2, q, q_vel) -q_vel(2)*psi_2*sin(q(2))*[2, 1; 1, 0];

    %Inverts matrix M
    M_inv = @(psi_1, psi_2, psi_3, q)inv(M(psi_1, psi_2, psi_3, q));
    
    %computes GAMMA Matrix on rhs of torque equation (WITHOUT SIN FACTOR),
    GAMMA = @(q_vel)[-q_vel(2), -sum(q_vel); q_vel(1), 0]; 

    delta_GAMMA = @(psi_2, q, q_vel, q_acc) q_vel(2)*psi_2*cos(q(2))*GAMMA(q_vel)...
        + psi_2*sin(q(2))*GAMMA(q_acc);

    JERK = @(tau_dot, curr_delta_M, curr_Gamma, curr_delta_Gamma, q, q_vel, q_acc)...
        tau_dot - (curr_delta_M + curr_Gamma)*q_acc - curr_delta_Gamma*q_vel;

    %Computes Torque using GAMMA and M matrix following equation in handout.
    Tao = @(psi_1, psi_2, psi_3, q, q_vel, q_acc)(M(psi_1, psi_2, psi_3, q) ...
    * q_acc) + (psi_2 * sin(q(2)) * GAMMA(q_vel) * q_vel);

    q = s(1:2);
    q_vel = s(3:4);
    q_acc = s(5:6);

    curr_torque = Tao(psi(1), psi(2), psi(3), q, q_vel, q_acc);
    curr_delta_M = delta_M(psi(2), q, q_vel);
    curr_delta_Gamma = delta_GAMMA(psi(2), q, q_vel, q_acc);
    curr_Gamma = psi(2) * sin(q(2)) * GAMMA(q_vel);
    torque_dot = a - (zeta * curr_torque);

    jerk_component = JERK(torque_dot, curr_delta_M,...
    curr_Gamma, curr_delta_Gamma, q, q_vel, q_acc);
    q_jerk = M_inv(psi(1), psi(2), psi(3), q) * jerk_component;

    q = q + Dt * q_vel;
    q_vel = q_vel + Dt * q_acc;
    q_acc = q_acc + Dt * q_jerk;

    q = max(q_min, min(q_max, q));
    q_vel = q_vel - q_vel.*( (q == q_max).*(q_vel > 0) + (q == q_min).*(q_vel < 0));

    s_next = [q; q_vel; q_acc];

end  % function test_policy
