% Laster inn dataene og konverterer tid fra ms til sekunder
Treningsdata = readtable('Treningsdata', 'Delimiter', ',', 'ReadVariableNames', false);
Treningsdata.Properties.VariableNames = {'Time_ms', 'PWM', 'Angle_deg'};
Treningsdata.Time_s = Treningsdata.Time_ms / 1000;

Valideringsdata = readtable('Valideringsdata', 'Delimiter', ',', 'ReadVariableNames', false);
Valideringsdata.Properties.VariableNames = {'Time_ms', 'PWM', 'Angle_deg'};
Valideringsdata.Time_s = Valideringsdata.Time_ms / 1000;

% Beregn sampletid (gjennomsnitt)
avg_sample_time_Treningsdata = mean(diff(Treningsdata.Time_s));
avg_sample_time_Valideringsdata = mean(diff(Valideringsdata.Time_s));


% Oppretter iddata-objekter

Treningsdata_id = iddata(Treningsdata.Angle_deg, Treningsdata.PWM, avg_sample_time_Treningsdata);
Valideringsdata_id = iddata(Valideringsdata.Angle_deg, Valideringsdata.PWM, avg_sample_time_Valideringsdata);

% Define the plant transfer function
G_original = tf(78.13, [1 0.9189 0.0531]);

% Normalize the transfer function to stabilize at 1
dc_gain = dcgain(G_original);
G = G_original / dc_gain;

% Automatically tune a PID controller
C = pidtune(G, 'PID');

% Create a closed-loop system
T = feedback(C*G, 1);

% Plot step responses in the same plot
figure;
hold on;
step(G, 'r'); % Open-loop step response in red
step(T, 'b'); % Closed-loop step response with PID in blue
hold off;

% Add title, labels, and legend
title('Step Response');
xlabel('Time');
ylabel('Amplitude');
legend('Open-Loop (G)', 'Closed-Loop with PID');
grid on;

% Display tuned PID parameters
disp('Tuned PID Controller:');
disp(C);


