% Laster inn data
Treningsdata = readtable('Treningsdata', 'Delimiter', ',', 'ReadVariableNames', false);
Treningsdata.Properties.VariableNames = {'Time_ms', 'PWM', 'Angle_deg'};
Treningsdata.Time_s = Treningsdata.Time_ms / 1000;
Valideringsdata = readtable('Valideringsdata', 'Delimiter', ',', 'ReadVariableNames', false);
Valideringsdata.Properties.VariableNames = {'Time_ms', 'PWM', 'Angle_deg'};
Valideringsdata.Time_s = Valideringsdata.Time_ms / 1000;

% Beregn sampletid
avg_sample_time_Treningsdata = mean(diff(Treningsdata.Time_s));
avg_sample_time_Valideringsdata = mean(diff(Valideringsdata.Time_s));

% Oppretter iddata-objekter
Treningsdata_id = iddata(Treningsdata.Angle_deg, Treningsdata.PWM, avg_sample_time_Treningsdata);
Valideringsdata_id = iddata(Valideringsdata.Angle_deg, Valideringsdata.PWM, avg_sample_time_Valideringsdata);

G_original = tf(78.13, [1 0.9189 0.0531]);
dc_gain = dcgain(G_original);
G = G_original / dc_gain;
C = pidtune(G, 'PID');
T = feedback(C*G, 1);

% Plot
figure;
hold on;
step(G, 'r');
step(T, 'b');
hold off;
title('Step Response');
xlabel('Time');
ylabel('Amplitude');
legend('Open-Loop (G)', 'Closed-Loop with PID');
grid on;
disp('Tuned PID Controller:');
disp(C);


