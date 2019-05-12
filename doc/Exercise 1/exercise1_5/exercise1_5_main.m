syms x y 
figure(4);
s1 = 1;
s2 = 4 ;
e21 = 1;
e12 = 0.5;
x = [0:0.01:10];
p  = raylpdf(x,1);
p1 = raylpdf(x,2);

subplot(1, 2, 1)
plot(x,p,x,p1)
legend({'?1','?2'},'Location','northeast')
title('Without penalty')
hold on

x = [0:0.01:10];
subplot(1, 2, 2);
p =  e12.*  raylpdf(x,1);
p1 = e21.* raylpdf(x,2);

plot(x,p,x,p1)
title('With penalty')
hold off
hold on
axis([0 10 0 0.7])
legend({'?1','?2'},'Location','northeast')
hold off