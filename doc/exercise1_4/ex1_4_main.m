clear all 
close all 
clc
mu1 = [3; 3];
sigma1 = [1.2 -0.4; -0.4 1.2];
mu2 = [6; 6] ; 
sigma2 = [1.2 0.4; 0.4 1.2];
syms  x1 x2
x = [x1; x2];
pw1 = [ 0.1 0.25 0.5 0.75 0.9];

figure(1) 
colorstring = 'kbgry';

for i=1:size(pw1,2)
    pw2(i) = 1- pw1(i);
    y = log(pw2(i)) + (-1/2)*(x-mu2)'*inv(sigma2)*(x-mu2) == log(pw1(i))+ (-1/2)*(x-mu1)'*inv(sigma1)*(x-mu1);
    hold on
    ez1(i) = ezplot(y,[-3 14 -3 14]); 
    set(ez1(i),'color',colorstring(i));
   
end
 hold off
% 
DT = 0.01;
x1=[-3:DT:14]; %Horizontal axis 
x2=[-3:DT:14]; 
hold on
[X1,X2]=meshgrid(x1,x2); 
Y1 = mvnpdf([X1(:) X2(:)],mu1',sigma1); 
Y1R=reshape(Y1,length(x2),length(x1));

contour(x1,x2,Y1R,[.0001 .001 .01 .05:.1:.95 .99 .999 .9999])
grid on 
hold off
hold on
title('Contour of classes ?1 ?2');
[X1,X2]=meshgrid(x1,x2); 
Y1 = mvnpdf([X1(:) X2(:)],mu2',sigma2); 
Y1R=reshape(Y1,length(x2),length(x1));

contour(x1,x2,Y1R,[.0001 .001 .01 .05:.1:.95 .99 .999 .9999]) 
hold off

