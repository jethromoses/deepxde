%Postprocessing the 2D heat transfer equation results

clc
close all
clear all

%load the data
test = importdata('../save_data/heat_transfer_2d_test.dat');
% loss = importdata('../save_data/Burgers_loss.dat');


x = test(:,1);
x = reshape(x, [100, 100]);
y = test(:,2);
y = reshape(y, [100, 100]);
y_pred = test(:,4);
y_pred = reshape(y_pred, [100, 100]);



figure(1)
hold on
box on
set(gca,'FontSize', 30)
set(gcf, 'Position',  [100, 100, 800, 600])
contourf(x,y,y_pred);
xlabel('x')
ylabel('y')
% caxis([0 1])
colorbar
hold off

figure(2)
hold on
box on
set(gca,'FontSize', 30)
set(gcf, 'Position',  [100, 100, 800, 600])
contour(x,y,y_pred,20);
xlabel('x')
ylabel('y')
% caxis([0 1])
colorbar
hold off
