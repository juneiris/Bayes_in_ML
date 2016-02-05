clear all;
data1=load('/Users/xuchuwen/Downloads/data_matlab/data1.mat');
x=(data1.X)';
y=data1.y;
z=data1.z;
N=size(x,2);
d=size(x,1);


%initialize
T=500;
a0=10^(-16);
b0=10^(-16);
e0=1;
f0=1;
mu=eye(d,1);
sigma=eye(d,d);

% sum1=zeros(d,d);
% sum2=zeros(d,1);
% sum=0;

Ealpha=ones(d,1);
vof=zeros(1,T);
for t=1:T
    sum1=zeros(d,d);
    sum2=zeros(d,1);
    summ=0;
    %update qalpha
    for k=1:d
        a=a0+1/2;
        b(k)=b0+0.5*(mu(k)^2+sigma(k,k));
        Ealpha(k)=a/b(k);
    end
    
    %update qlambda
    e=e0+N/2;
    for i=1:N
        summ=summ+(y(i)-x(:,i)'*mu)^2+x(:,i)'*sigma*x(:,i);
    end
    f=f0+0.5*summ;
    Elambda=e/f;

    %update qw
    for i=1:N
        sum1=sum1+x(:,i)*x(:,i)'; 
        sum2=sum2+x(:,i)*y(i);
    end
    sigma=diag(Ealpha)+(e/f)*sum1;
    mu=sigma\((e/f)*sum2);
    sigma=pinv(sigma);
    
    %compute log(det(sigma))
    [u,dd]=eig(sigma);
    logdetsig=sum(log(diag(dd)));
    
    %compute variational objective function
    Epy=0;Epalpha=0;Epw=0;Epw1=0;Epw2=0;Epalpha=0;Eplambda=0;
    Eqalpha=0;Eqw=0;Eqlambda=0;
    
    for i=1:N
        Epy=Epy+(y(i)-x(:,i)'*mu)^2+x(:,i)'*sigma*x(:,i);
    end
    Epy=(N/2)*(psi(e)-log(f)-log(2*pi))-0.5*(e/f)*Epy;
    
    for k=1:d
        Epw1=Epw1+psi(a)-log(b(k));
        Epalpha=Epalpha+a0*log(b0)-gammaln(a0)+(a0-1)*(psi(a)-log(b(k)))-b0*a/b(k);
        Eqalpha=Eqalpha+log(b(k))-gammaln(a)+(a-1)*psi(a)-a;
        %Eqw=Eqw+log(sigma(k,k));
    end
    Epw2=diag(Ealpha)*(mu*mu'+sigma);
    Epw=0.5*Epw1-0.5*trace(Epw2)-(d/2)*log(2*pi);
    
    Eplambda=e0*log(f0)-gammaln(e0)+(e0-1)*(psi(e)-log(f))-f0*e/f;
    
    Epall=Epy+Epw+Epalpha+Eplambda;
    
    %Eqw=(-d/2)*log(2*pi)-0.5*Eqw-(d/2);
    Eqw=(-d/2)*log(2*pi)-0.5*logdetsig-(d/2);
    
    Eqlambda=log(f)-gammaln(e)+(e-1)*psi(e)-e;
    
    vof(t)=Epall-Eqalpha-Eqw-Eqlambda;
    
end


%plot variational objective function
figure(1)
plot(1:T,vof)
title('variational objective function')
xlabel('t')
ylabel('vof')

%plot 1/Ealpha
figure(2)
plotEa=1./Ealpha;
stem(1:d,plotEa);
title('1/E[alpha]');
xlabel('k');
ylabel('1/E[alpha]');

invEl=1/Elambda


%plot z
figure(3)
yhat=zeros(N,1);
z2=zeros(N,1);
for i=1:N
    yhat(i)=x(:,i)'*mu;
    z2(i)=10*sinc(z(i));   
end
i=1:N;
plot(z(i),yhat(i));
hold on;
scatter(z(i),y(i));
plot(z(i),z2(i),'r');
legend('yhat','scatter plot','10*sinc(z)');





