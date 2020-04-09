clear all;
%close all;
Z0 = 377;
c  = 299792458;

%current      = h5read('./example_current_profile.h5','/current');
%energy_eV    = h5read('./example_current_profile.h5','/energy_eV');
%time_profile = h5read('./example_current_profile.h5','/time_profile');

load("./example_current_profile.mat")

t = 1e12*(time_profile-time_profile(1)); %[ps]
I = current;

Q = trapz(t,I);
I = I/Q;

%bunch profile with new s coordinate
ss  = c*t*1e-12; %m
deltas = 0.1e-6;
sb  = 0:deltas:max(ss);
rho = spline(ss, I/trapz(ss,I), sb);

figure(1);
subplot(3,1,2);
plot(1e6*sb,1e-2*rho,'Linewidth', 3);
set(gca, 'FontSize', 20, 'Fontname', 'Times');
xlabel('s [\mum]');
ylabel('I [a.u.]');
hold on

%s-coordinate for wake function 
s = 0:deltas:max(sb);

%wake function
gap   = 4.0e-3; %m
a     = gap/2;
p     = 0.500e-3; %m period
delta = 0.250e-3; %m corrugation depth
g     = 0.250e-3; %m longitudinal gap

alfa = 1-0.465*sqrt(g/p)-0.07*g/p;
s0r  = a^2*delta/2/pi/alfa^2/p^2; %
s0d  = s0r*(15/14)^2;
wxd  = Z0*c/4/pi*pi^4/16/a^4*s0d*(1-(1+sqrt(s/s0d)).*exp(-sqrt(s/s0d))); 
save "-mat7-binary" "wxd.mat" wxd s
w1   = wxd;

figure(1);
subplot(3,1,1);
plot(1e6*s,1e-15*w1, 'Linewidth', 2);
set(gca, 'FontSize', 20, 'Fontname', 'Times');
grid on
xlabel('s [\mum]');
ylabel('wxd [MV/nC/m/m]');
title('Wake function', 'FontSize', 16, 'Fontname', 'Times');
hold on

%convolution

Wb_pass = deltas*conv(rho,w1); %m
Nb      = length(rho);
Wb      = Wb_pass(1:Nb); %m

Kt = trapz(sb,Wb.*rho) %Mv/nC/m/m

disp(['Transverse kick factor: ' num2str(Kt) ' MV/nC/m/m']);

figure(1);
subplot(3,1,2);
hold on
plot(1e6*sb, 1e-15*Wb, 'Color', [0 0.5 0], 'Linewidth', 3);
grid on
set(gca, 'FontSize', 20, 'Fontname', 'Times');
xlabel('s [\mum]', 'FontSize', 20, 'Fontname', 'Times');
ylabel('W_{xd}(s) [MV/nC/m/m]', 'FontSize', 20, 'Fontname', 'Times' );
title('Wake potential and bunch profile', 'FontSize', 16, 'Fontname', 'Times');


Q      = 1e-12*Q;  % charge in C
optics = 11; %m
Lguide  = 1; %m
E      = energy_eV; %beam energy in eV

Xc = Kt * Q * optics * Lguide/E % centroid MV/nC/offset[m]

Xoffset = 1e-3*(-1:0.1:1); %m

figure(1);
subplot(3,1,3);
plot(1000*Xoffset, 1e6*(Xc*Xoffset), 'Linewidth', 3 );
grid on
set(gca, 'FontSize', 20, 'Fontname', 'Times');
xlabel('\DeltaX [mm]', 'FontSize', 20, 'Fontname', 'Times');
ylabel('\langleX\rangle [\mum]', 'FontSize', 20, 'Fontname', 'Times' );
title('Displacement', 'FontSize', 16, 'Fontname', 'Times');
%axis([-1 1 -1 1.5]);


% sb = sb';
% save('wakepot.mat', 'sb', 'rho');

% A_str = ['Wb',num2str(Nm), '=Wb'];
% eval(A_str);
% disp(['Transverse kick factor: ' num2str(Kt) ' MV/nC/m/m']);
% save('wakepot.mat', ['Wb',num2str(Nm)], '-append');
