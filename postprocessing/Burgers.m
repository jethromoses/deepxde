%Postprocessing the Burgers equation results

clc
close all
clear all

%load the data
test = importdata('../save_data/Burgers_test.dat');
loss = importdata('../save_data/Burgers_loss.dat');


x = test(:,1);
x = reshape(x, [256, 100]);
t = test(:,2);
t = reshape(t, [256, 100]);
y_true = test(:,3);
y_true = reshape(y_true, [256, 100]);
y_pred = test(:,4);
y_pred = reshape(y_pred, [256, 100]);



figure(1)
hold on
box on
set(gca,'FontSize', 30)
set(gcf, 'Position',  [100, 100, 800, 600])
contourf(t,x,y_pred);
xlabel('time')
ylabel('x')
caxis([-1 1])
colorbar
hold off


figure(2)
hold on
box on
set(gca,'FontSize', 30)
set(gcf, 'Position',  [100, 100, 800, 600])
contourf(t,x,y_true);
xlabel('time')
ylabel('x')
caxis([-1 1])
colorbar
hold off
