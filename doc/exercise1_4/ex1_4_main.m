clear all 
close all 

mu1 = [3 3];  
sigma1 = [1.2 -0.4; -0.4 1.2];
mu2 = [6 6];  
sigma2 = [1.2 0.4; 0.4 1.2];

figure(1)
syms x y 
DT = 0.01;
pw1 = [ 0.1 0.25 0.5 0.75 0.9];


for i = 1:size(pw1,2)
    pw2 = 1- pw1(i);
    y = - x + 9 - 0.4*log(pw1(i)/pw2)
    hold on
    ezplot(y,[-4 12 -4 12]); 
end
hold off
x1=[-20:DT:20]; %Horizontal axis 
x2=[-20:DT:20]; 
hold on
[X1,X2]=meshgrid(x1,x2); 

Y1 = mvnpdf([X1(:) X2(:)],mu1,sigma1); 
Y1R=reshape(Y1,length(x2),length(x1)); 
contour(x1,x2,Y1R,[.0001 .001 .01 .05:.1:.95 .99 .999 .9999]) 
grid on 
hold off

hold on
[X1,X2]=meshgrid(x1,x2); 

Y1 = mvnpdf([X1(:) X2(:)],mu2,sigma2); 
Y1R=reshape(Y1,length(x2),length(x1)); 
contour(x1,x2,Y1R,[.0001 .001 .01 .05:.1:.95 .99 .999 .9999]) 
hold off
grid on 