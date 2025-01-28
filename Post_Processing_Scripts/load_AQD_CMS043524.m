clear all
close all
% Enter input /directory/ and fileName root without file extension
inputDir  = 'C:\Users\lwlav\OneDrive\Documents\Summer 2024 CHAZ\Data\ADCP_05202024_unconverted';
inputFile = 'CMS52002';
fileName  = [inputDir,'/',inputFile];
% Enter raw output /directory/ and fileName without .mat
outputDir = 'C:\Users\lwlav\OneDrive\Documents\Summer 2024 CHAZ\Data';
outputName= [inputFile,'_raw'];
% Enter processed output fileName without .mat
L0Name  = [inputFile,'_L0'];
% Enter path to save figures
figDir = [inputDir];
if ~exist(figDir,'dir'), eval(['!mkdir -p ',figDir]), end
%
% Enter time-offset (UTC->EDT) tos = -4 hours
tos = -4;
%% load header data
%% data for each field starts at column 39 or 40
hdrFile = sprintf('%s/%s.hdr', inputDir,inputFile);
fid = fopen(hdrFile);% open file
while ~feof(fid)
    %grab line
    line = fgetl(fid);
    %some skipped lines are very short
    if length(line)<26
        continue
    end
    %
    % each line has a field name and value
    string = deblank(line(1:26));
    value = line(39:end);
    %
    % select pertinent fields and allocate variables
    if     strcmp(string,'Number of measurements')
        nsamples = str2num(value);
    elseif strcmp(string,'Number of checksum errors')
        nerrors = str2num(value);
    elseif strcmp(string,'Measurement/Burst interval')
        i = strfind(value,'sec');
        dt = str2num(value(1:i-1));
    elseif strcmp(string,'Number of cells')
        nbins = str2num(value);
    elseif strcmp(string,'Cell size')
        i = strfind(value,'cm');
        cff=100;
        if isempty(i)
            i = strfind(value,'mm');
            cff=1000;
        end
        binsize = str2num(value(1:i-1))/cff;% cm-->m
    elseif strcmp(string,'Blanking distance')
        i = strfind(value,'m');
        blank = str2num(value(1:i-1));
    elseif strcmp(string,'Coordinate system')
        coords = value(1:3);
    elseif strcmp(string,'Serial number') & ~exist('sn','var')
        sn = deblank(value);
    end
clear line string value i
end

meta_data = struct('SN',sn,'Nsamples',nsamples,'Nerrors',nerrors,'dt',dt, ...
                   'Nbins',nbins,'binSize',binsize,'blank',blank,'coords',coords);
fclose(fid);
%
%% Extracts date, temp, pressure, heading, pitch, roll
senFile = sprintf('%s/%s.sen', inputDir,inputFile);
fid = fopen(senFile,'r');
A = textscan(fid,'%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %*[^\n]');
fclose(fid);
A1 = [A{:,1}, A{:,2}, A{:,3}, A{:,4}, A{:,5}, A{:,6}, A{:,7}, A{:,8}, A{:,9}, A{:,10}, A{:,11}, A{:,12} A{:,13}, A{:,14}, A{:,15} A{:,16} A{:,17}];
date = datenum(A1(:,3),A1(:,1),A1(:,2),A1(:,4),A1(:,5),A1(:,6))+tos/24;
volt = A1(:,11);
sspeed = A1(:,12);
head = A1(:,13);pitch = A1(:,14);roll = A1(:,15);
pres = A1(:,16);
temp = A1(:,17);
clear A A1
%
A.config= meta_data;
A.date  = date;A.volt = volt;
A.seconds= (date-date(1))*86400;
A.sspeed= sspeed;
A.head  = head;A.pitch = pitch;A.roll = roll;
A.pres  = pres;A.temp = temp;
A.fname = fileName;
%
%% load beam amplitudes and velocities (may be in beam coords or ENU, see A.config)
a1 = load(strcat(fileName,'.a1'));
[Na,Ma] = size(a1);
if Ma>nbins% sometimes there are extra columns (beam#, ensemble#)
    bin1 = Ma-nbins+1;
else
    bin1 = 1;
end
a2 = load(strcat(fileName,'.a2'));
a3 = load(strcat(fileName,'.a3'));
A.a1 = a1(:,bin1:end);
A.a2 = a2(:,bin1:end);
A.a3 = a3(:,bin1:end);
%
v1 = load(strcat(fileName,'.v1'));
v2 = load(strcat(fileName,'.v2'));
v3 = load(strcat(fileName,'.v3'));
A.v1 = v1(:,bin1:end);
A.v2 = v2(:,bin1:end);
A.v3 = v3(:,bin1:end);
%
A.dbins = blank:binsize:((nbins-1)*binsize)+blank;
% 
save([outputDir,'/',outputName,'.mat'],'-struct','A')
%
% plot some stuff
disp('pick 2 points bounding when out of water for ATM pressure offset')
plot(pres)
l = ginput(2);
A.pressureOffset = mean(pres(round(l(1,1)):round(l(2,1))));
%
% now trim the data to when it was in the water
disp('pick start/end points of deployment')
l = ginput(2);
valid = [round(l(1,1)):round(l(2,1))]';
vars  = {'date','volt','seconds','sspeed','head','pitch','roll','pres','temp','a1','a2','a3','v1','v2','v3'};
for jj = 1:length(vars)
    eval(['A.',vars{jj},' = A.',vars{jj},'(valid,:);'])
end
nsamples = length(valid);
%
A.maxRange = (A.pres-A.pressureOffset).*cosd(20)-1*binsize;
dum1 = A.maxRange.*ones(1,nbins);
dum2 = ones(nsamples,1)*A.dbins;
qcFlag0  =  (dum2<=dum1);
A.qcFlag =  (dum2<=dum1) & min(A.a1,min(A.a2,A.a3))>130;
%
time = datetime(A.date,'convertFrom','datenum');
% make a few quick plots
fig0 = figure;
ax1 = subplot(2,1,1);
plot(time,A.temp)
ylabel(ax1,'$T$ [$^\circ$]','interpreter','latex')
set(ax1,'xticklabel','','ticklabelinterpreter','latex','tickdir','out')
ax2 = subplot(2,1,2);
plot(time,A.pres)
ylabel(ax2,'$P$ [m]','interpreter','latex')
xlabel(ax2,'time [s]','interpreter','latex')
set(ax2,'ticklabelinterpreter','latex','tickdir','out')
figName = [figDir,'/',inputFile,'_temp_pres.png'];
exportgraphics(fig0,figName)
%
%
% quick convolution running mean filter
np1 = round(0.3/A.config.binSize);% 30 cm vertical 
np2 = 61;% 1 minute for 1Hz data
f1 = hamming(np1);f1 = f1./sum(f1);
f2 = hamming(np2);f2 = f2./sum(f2);
A1 = conv2(f1,f2,A.a1','same');
fig1 = figure;
ax1 = subplot(3,1,1);
imagesc(time,A.dbins',A1.*qcFlag0'),caxis([100 180]),colorbar
text(time(1),max(A.maxRange,[],"omitnan"),'East')
set(ax1,'ydir','normal','ticklabelinterpreter','latex','ylim',[0 max(A.maxRange,[],"omitnan")])
A2 = conv2(f1,f2,A.a2','same');
ax2 = subplot(3,1,2);
imagesc(time,A.dbins',A2.*qcFlag0'),caxis([100 180]),colorbar
text(time(1),max(A.maxRange,[],"omitnan"),'North')
ylabel('mab','interpreter','latex')
set(ax2,'ydir','normal','ticklabelinterpreter','latex','ylim',[0 max(A.maxRange,[],"omitnan")])
A3 = conv2(f1,f2,A.a3','same');
ax3 = subplot(3,1,3);
imagesc(time,A.dbins',A3.*qcFlag0'),caxis([100 180]),colorbar
text(time(1),max(A.maxRange,[],"omitnan"),'Up')
set(ax3,'ydir','normal','ticklabelinterpreter','latex','ylim',[0 max(A.maxRange,[],"omitnan")])
xlabel('time [s]','interpreter','latex')
figName = [figDir,'/',inputFile,'_amplitude.png'];
exportgraphics(fig1,figName)
%
% plot currents
V1 = conv2(f1,f2,A.v1','same');
fig2 = figure;
ax1 = subplot(3,1,1);
imagesc(time,A.dbins',V1.*A.qcFlag'),caxis([-1 1]),colorbar
text(time(1),max(A.maxRange,[],"omitnan"),'East')
V2 = conv2(f1,f2,A.v2','same');
set(ax1,'ydir','normal','ticklabelinterpreter','latex','ylim',[0 max(A.maxRange,[],"omitnan")])
ax2 = subplot(3,1,2);
imagesc(time,A.dbins',V2.*A.qcFlag'),caxis([-0.5 0.5]),colorbar
text(time(1),max(A.maxRange,[],"omitnan"),'North')
set(ax2,'ydir','normal','ticklabelinterpreter','latex','ylim',[0 max(A.maxRange,[],"omitnan")])
ylabel('mab','interpreter','latex')
V3 = conv2(f1,f2,A.v3','same');
ax3 = subplot(3,1,3);
imagesc(time,A.dbins',V3.*A.qcFlag'),caxis([-0.25 0.25]),colorbar
text(time(1),max(A.maxRange,[],"omitnan"),'Up')
set(ax3,'ydir','normal','ticklabelinterpreter','latex','ylim',[0 max(A.maxRange,[],"omitnan")])
xlabel('time [s]','interpreter','latex')
figName = [figDir,'/',inputFile,'_velocity.png'];
exportgraphics(fig2,figName)
%
% get the time-averaged current.
% Note, this is not in depth normalized (sigma) coordinates.
% first, nan any values that don't pass QC.
V1(~A.qcFlag')=nan;
V2(~A.qcFlag')=nan;
V3(~A.qcFlag')=nan;
flag = sum(A.qcFlag,1) > 0.70*nsamples;
% Uz = time averegd; U = depth & time averaged
Uz = mean(V1,2,"omitnan"); U = mean(Uz,"omitnan"); Uz(~flag)=nan;
Vz = mean(V2,2,"omitnan"); V = mean(Vz,"omitnan"); Vz(~flag)=nan;
Wz = mean(V3,2,"omitnan"); W = mean(Wz,"omitnan"); Wz(~flag)=nan;
%
fig3 = figure;
plot(Uz,A.dbins,'.-r',Vz,A.dbins,'.-b',Wz,A.dbins,'.-k')
hh = legend('East','North','Up');
set(hh,'fontsize',9)
xlabel('m/s','interpreter','latex')
ylabel('mab','interpreter','latex')
set(gca,'ticklabelinterpreter','latex','tickdir','out')
figName = [figDir,'/',inputFile,'_mean_velocity.png'];
exportgraphics(fig3,figName)

A.VelEast = V1;
A.VelNorth= V2;
A.VelUp   = V3;

A.VelEastAvg = Uz;
A.VelNorthAvg= Vz;
A.VelUpAvg   = Wz;

save([outputDir,'/',L0Name,'.mat'],'-struct','A')