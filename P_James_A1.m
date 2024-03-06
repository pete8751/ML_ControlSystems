clear variables;
clc;
clf;

%DEFINING HELPER FUNCTIONS --

%Note that in the following functions q/q_vel, etc are just param names. They wouldn't be equivalent
%to the actual q/q_vel, etc, but rather whatever is passed in. You can
%check this by changing psi/zeta estimates and noting that outcome changes.
Hurwitz = @(q, q_vel, q_acc, q_star, q_star_vel) (-45 * q_acc) + (-675 * (q_vel - q_star_vel)) + (-3375 * (q - q_star));

%computes M Matrix on rhs of torque equation, (first and second derivative of q)
M = @(psi_1, psi_2, psi_3, q)[psi_1 + (2 * psi_2 * cos(q(2))), psi_3 + (psi_2 * cos(q(2))); 
    psi_3 + (psi_2 * cos(q(2))), psi_3];

%Inverts matrix M
M_inv = @(psi_1, psi_2, psi_3, q)inv(M(psi_1, psi_2, psi_3, q));

%computes GAMMA Matrix on rhs of torque equation,
GAMMA = @(q, q_vel)[-q_vel(2), -sum(q_vel); q_vel(1), 0] * q_vel; 

%Computes Torque using GAMMA and M matrix following equation in handout.
Tao = @(psi_1, psi_2, psi_3, q, q_vel, q_acc)(M(psi_1, psi_2, psi_3, q) ...
    * q_acc) + (psi_2 * sin(q(2)) * GAMMA(q, q_vel));

%computes q_acc by inversing torque equation using current torque and q and q_vel
ACC = @(psi_1, psi_2, psi_3, q, q_vel, torque)M_inv(psi_1, psi_2, psi_3, q)...
    * (torque - ((psi_2 * sin(q(2)) * GAMMA(q, q_vel)))); 

% SETTING PARAMETERS --
Dt = 0.01;  % time step in s
dur = 10;  % duration of run in s
step = 0;  % # time steps since simulation began

zeta = 0.75;
[psi_1, psi_2, psi_3] = deal(1.88, 0.6, 0.48);

q_min = [-2; 0]; % max and min joint constraints.
q_max = [1.5; 2.5];

% INITIALIZE --
t = 0;
a = [0; 0]; %action values

q = [0; 0]; % Agent joint-angle vectors [shoulder, elbow]
q_vel = [0; 0];
q_acc = [0; 0];
torque = [0; 0];

q_est = [0; 0]; % Agent joint-angle vector estimates [shoulder, elbow]
q_est_vel = [0; 0];
q_est_acc = [0; 0];
torque_est = [0; 0];

psi_1_est = 1.88; %Agent Parameter Estimates
psi_2_est = 0.6;
psi_3_est = 0.48;
zeta_est = 0.75;

q_star = [0; 0];  % target joint-angle vectors [shoulder, elbow]
q_star_vel = [0; 0]; 

% Set up graphics
DATA = zeros(9, 1 + floor(dur/Dt));  % allocate memory
i_gp = 1;  % index of graph pts
DATA(:, i_gp) = [t; q; q_star; a; q_star_vel];

% Run time loop
for t = 0:Dt:dur %This loop iterates over time steps from 0 to dur with a time step of Dt

    if mod(t, 1) == 0
        %Calculating new random position and velocity values for target.
        new_shoulder_angle = (rand * 3.5) + q_min(1);
        new_elbow_angle = (rand * 2.5) + q_min(2);
        
        %the min/max velocity range for shoulder/elbow_glide (so we don't leave bounds).
        bound_shoulder_glide_vel = [q_min(1) - new_shoulder_angle; q_max(1) - new_shoulder_angle];
        bound_elbow_glide_vel = [q_min(2) - new_elbow_angle; q_max(2) - new_elbow_angle]; 
        
        %Assigning random position/velocities within given ranges for target
        new_shoulder_angle_vel = rand * (bound_shoulder_glide_vel(2) - bound_shoulder_glide_vel(1)) +  bound_shoulder_glide_vel(1);
        new_elbow_angle_vel = rand * (bound_elbow_glide_vel(2) - bound_elbow_glide_vel(1)) +  bound_elbow_glide_vel(1);
        q_star = [new_shoulder_angle; new_elbow_angle];
        q_star_vel = [new_shoulder_angle_vel; new_elbow_angle_vel];
        
        % resetting agent estimates to correct value 
        q_est = q;
        q_est_vel = q_vel;
        q_est_acc = q_acc;
        torque_est = torque;
        %Note q_est_acc doesn't need to be reset, as it is always
        %recalculated from the other estimates. I will just reset it
        %regardless, even though it gives a small underline warning.
    end

    % COMPUTE NEXT STATE
    
    %1. Agents estimated next state.

    %Action controls change in torque, which is proportional to
    %acceleration, acceleration controls change in velocity which controls
    %change in position.

    %Using third order hurwitz we choose desired change in acceleration 
    % that drives distance from target to 0
    delta_a_desired = Hurwitz(q_est, q_est_vel, q_est_acc, q_star, q_star_vel); %This was initially q_vel, but I changed this after the assignment
    %I lost points on the above, and because there were some unsmooth
    %parts. 

    %I now convert this change in acceleration to its equivalent value as a
    %change in torque. (we might have done this conversion 
    %inside the earlier polynomial, but this would have caused 
    %the formulae to become unreadable, so I just did it after finding 
    %delta_a_desired.)

    %Finding delta_torque that gives equivalent change in acceleration
    torque_desired = Tao(psi_1_est, psi_2_est, psi_3_est, q_est, q_est_vel, q_est_acc + delta_a_desired);
    delta_torque_desired = torque_desired - torque_est;
    %Choosing action that gives desired delta_torque (based on diff equation).
    a = (zeta_est * torque_est) + (delta_torque_desired); 

    %Now agent calculates future state based on action:
    torque_dot_est = (a - (zeta_est * torque_est));
    % Calculating next state torque estimate.
    torque_est = torque_est + Dt * torque_dot_est;
    % Calculating acceleration estimate by reversing torque equation
    q_est_acc = ACC(psi_1_est, psi_2_est, psi_3_est, q_est, q_est_vel, torque_est);
    % Updating Velocity estimate.
    q_est_vel = q_est_vel + Dt * q_est_acc;
    % Updating Position estimate.
    q_est = q_est + Dt * q_est_vel;

    %Keep agent estimates in bounds:
    q_est = max(q_min, min(q_max, q_est));
    q_est_vel = q_est_vel - q_est_vel.*( (q_est == q_max).*(q_est_vel > 0) + (q_est == q_min).*(q_est_vel < 0));

    %2. Agent's actual next state (repeats above calculations, with correct parameter vals).
    torque_dot = (a - (zeta * torque));
    torque = torque + Dt * torque_dot;
    q_acc = ACC(psi_1, psi_2, psi_3, q, q_vel, torque);
    q_vel = q_vel + Dt * q_acc;
    q = q + Dt * q_vel;

    q = max(q_min, min(q_max, q));
    q_vel = q_vel - q_vel.*( (q == q_max).*(q_vel > 0) + (q == q_min).*(q_vel < 0));
    
    %Updating target position.
    q_star = q_star + Dt * q_star_vel;  

    % Record data for plotting
    i_gp = i_gp + 1;
    DATA(:, i_gp) = [t; q_est; q_star; a; a];

end  % t
DATA = DATA(:, 1:i_gp);

close all;

% Plot
figure;

% First panel
subplot(2, 1, 1);
plot(DATA(1, :), DATA(2, :), 'r');  % Plot first component of q*
hold on;
plot(DATA(1, :), DATA(3, :), 'b');  % Plot second component of q*
plot(DATA(1, :), DATA(4, :), ':', 'LineWidth', 0.5, 'Color', 'red');  % Plot first component of q*
plot(DATA(1, :), DATA(5, :), ':', 'LineWidth', 1, 'Color', 'blue');  % Plot second component of q*
title('Agent and Target Position');
xlim([0, dur]);
ylim(1.05*[min(q_min), max(q_max)]);
ylabel('q');
xlabel('t');
legend('q_1', 'q_2', 'q_1*', 'q_2*');

% Second panel
subplot(2, 1, 2);
plot(DATA(1, :), DATA(6, :), 'r');  % Plot first component of action
hold on;
plot(DATA(1, :), DATA(7, :), 'b');  % Plot second component of action
title('Action');
xlim([0, dur]);
min_y_lim = min(min(DATA(6:7, :)));
max_y_lim = max(max(DATA(6:7, :)));
ylim(1.05*[min_y_lim, max_y_lim]);
ylabel('a');
xlabel('t');
legend('a_1', 'a_2');

set(gca, 'TickLength', [0, 0]);
set(gcf, 'Name', 'Inertial time opt', 'NumberTitle', 'off');