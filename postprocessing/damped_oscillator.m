%Postprocessing the damped oscillator results

clc
close all
clear all

%load the data
test = importdata('../save_data/damped_oscillator_test.dat');
% loss = importdata('../save_data/Burgers_loss.dat');


t = test.data(:,1);
y_true = test.data(:,2);
y_pred = test.data(:,3);

figure(1)
hold on
box on
set(gca,'FontSize', 30)
set(gcf, 'Position',  [100, 100, 800, 600])
plot(t,y_true,'Linewidth',2)
plot(t,y_pred,'--','Linewidth',2)
xlim([0 32])
ylim([-0.5 1])
xlabel('time')
ylabel('y')
legend('Analytical','PINN')