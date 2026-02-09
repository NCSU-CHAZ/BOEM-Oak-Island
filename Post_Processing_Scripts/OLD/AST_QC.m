%% AST QC via Rutten et al., 2024 Method
% BG 08/18/2025

% load data
path='/Volumes/kanarde/BOEM/deployment_1/Raw/S0_103080_mat/S103080A006_NCSU_15.mat'; % path to mat file
load(path); 

ast = Data.Burst_AltimeterDistanceAST;
press= Data.Burst_Pressure;
t   = Data.Burst_Time;
dt  = (t(2)-t(1))*86400;

% Detrend data
ast_detrended=detrend(ast,2); % quadratic
press_detrended=detrend(press,2); % quadratic

% Low pass filter data
cutoff_freq=0.01; % Hz 
Fs=4; % sampling frequency

ast_lowpass=lowpass(ast_detrended,cutoff_freq,Fs);
press_lowpass=lowpass(press_detrended,cutoff_freq,Fs);

% zero-meaned data
ast_zm=ast_detrended-ast_lowpass;
press_zm=press_detrended-press_lowpass;

% Plot data to check
figure(1); subplot(4,1,1); plot(t,ast,'Linewidth',2); hold on; plot(t,press,'Linewidth',2);
hold on; legend('AST Raw', 'Pressure Raw'); ylabel('meters'); ylim([6,12]); subplot(4,1,2); plot(t,ast_detrended,'Linewidth',2); 
hold on; plot(t,press_detrended,'Linewidth',2); hold on; legend('AST detrended', 'Pressure detrended');
hold on; ylabel('meters'); ylim([-3,3]);subplot(4,1,3); plot(t,ast_lowpass,'Linewidth',2); hold on;plot(t,press_lowpass,'Linewidth',2); 
hold on; legend('AST lowpass', 'Press lowpass'); ylabel('meters');ylim([-3,3]);
hold on; subplot(4,1,4); plot(t,ast_zm,'Linewidth',2); hold on; plot(t,press_zm,'Linewidth',2); hold on;
legend('AST zero-meaned','Press zero-meaned');ylabel('meters');xlabel('time');ylim([-3,3]);

% Calculate SD per hour burst and determine threshold
hr=3600*4; % 1 hour with 4 Hz sampling frequency

SD=zeros(length(t)/hr,1);

for i=1:length(t)/hr
    hr_start=(i-1)*hr+1;
    hr_end=i*hr;
    
    ast_hr=ast_zm(hr_start:hr_end);
    press_hr=press_zm(hr_start:hr_end);

    SD(i)=std(ast_hr-press_hr);
end

% plot SD
figure(2); plot(SD,'o','Linewidth',2,'Color','k'); ylabel('SD value'); xlabel('Hour'); ylim([0,0.25]);
hold on; yline(mean(SD),'r','Linewidth',2); hold on; legend('SD values', 'Mean SD');

% Apply threshold to flag data
thresh=mean(SD);
flagged_ast_SD = find(abs(ast_zm)>thresh); % store flagged data
flagged_ast_SD_percent=(length(flagged_ast_SD)/length(ast_zm))*100;
flagged_press_SD= find(abs(press_zm)>thresh); % store flagged data
flagged_press_SD_percent=(length(flagged_press_SD)/length(press_zm))*100;

% Plot flagged points
figure(3); subplot(1,2,1); plot(ast_zm,'Linewidth',2); hold on; plot(flagged_ast_SD, ast_zm(flagged_ast_SD),'r.');
hold on; legend('AST zero mean', 'AST flagged by SD thresh'); ylabel('meters'); ylim([-1.5,1.5]);hold on;
subplot(1,2,2); plot(press_zm,'Linewidth',2); hold on; plot(flagged_press_SD, press_zm(flagged_press_SD),'r.');
hold on; legend('Press zero mean', 'Press flagged by SD thresh'); ylabel('meters'); xlabel('time');ylim([-1.5,1.5]);

% apply SD flag to data (replace with Nans)
ast_zm(flagged_ast_SD)=NaN;
press_zm(flagged_press_SD)=NaN;

% Calculate universal threshold
lambda=zeros(length(t)/hr,1);

for i=1:length(t)/hr
    hr_start=(i-1)*hr+1;
    hr_end=i*hr;
    
    ast_hr=ast_zm(hr_start:hr_end);
    press_hr=press_zm(hr_start:hr_end);

    lambda(i)=(sqrt(2*log(length(ast_hr)*SD(i))));
    lambda_times_SD(i)=lambda(i)*SD(i);
end

% plot universal threshold
figure(4); plot(lambda_times_SD,'o','Linewidth',2,'Color','k'); ylabel('Universal thresh value'); xlabel('Hour'); ylim([0,1]);
hold on; yline(mean(lambda_times_SD),'r','Linewidth',2); hold on; legend('Universal thresh values', 'Mean thresh');

% Apply threshold to flag data
thresh=mean(lambda_times_SD);
flagged_ast_ut = find(abs(ast_zm)>thresh); % store flagged data
flagged_ast_ut_percent=(length(flagged_ast_ut)/length(ast_zm))*100;
flagged_press_ut= find(abs(press_zm)>thresh); % store flagged data
flagged_press_ut_percent=(length(flagged_press_ut)/length(press_zm))*100;

% Plot flagged points
figure(3); subplot(1,2,1); plot(ast_zm,'Linewidth',2); hold on; plot(flagged_ast_ut, ast_zm(flagged_ast_ut),'r.');
hold on; legend('AST zero mean - QC', 'AST flagged by ut thresh'); ylabel('meters'); ylim([-0.3,0.3]);hold on;
subplot(1,2,2); plot(press_zm,'Linewidth',2); hold on; plot(flagged_press_ut, press_zm(flagged_press_ut),'r.');
hold on; legend('Press zero mean - QC', 'Press flagged by ut thresh'); ylabel('meters'); xlabel('time');ylim([-0.3,0.3]);

% apply SD flag to data (replace with Nans)
ast_zm(flagged_ast_ut)=NaN;
press_zm(flagged_press_ut)=NaN;