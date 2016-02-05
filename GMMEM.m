clear all;
data=load('/Users/xuchuwen/Downloads/data.mat');
x=data.X;
N=size(x,2);

%initialize
T=100;
K=8;
mu=zeros(size(x,1),K);
sigma=zeros(size(x,1),size(x,1),K);
pi=zeros(K,1);
lnpx=zeros(T,1);
%mu=eye(2,K);
for j=1:K
   %mu(:,j)=zeros(size(x,1),1);
   mu(:,j)=j*(-1)^j*ones(size(x,1),1);
   
   sigma(:,:,j)=eye(size(x,1),size(x,1));
   pi(j)=1/K; 
end
%mu=[-2,1;-2,4];

%pi=[0.2;0.8];
%sigma(:,:,1)=eye(size(x,1),size(x,1));
%sigma(:,:,2)=[2,1;1,2];

for t=1:T

    %E step
    for i=1:N
        summ=0;
        for j=1:K
            summ=summ+pi(j)*mvnpdf(x(:,i),mu(:,j),sigma(:,:,j));
            phi(i,j)=pi(j)*mvnpdf(x(:,i),mu(:,j),sigma(:,:,j));
            
        end
        phi(i,:)=phi(i,:)/summ;
%         for k=1:K
%             pp(i,k)=mvnpdf(x(:,i),mu(:,k),sigma(:,:,k));
%             %phi(i,k)=(pi(k)*mvnpdf(x(:,i),mu(:,k),sigma(:,:,k)))/summ;
%             phi(i,k)=pi(k)*pp(i,k);
%         end
        
       %phi(i,:)=phi(i,:)/summ; 
    end
    
    
    %M step
    for j=1:K
        nj=sum(phi(:,j));
        muj=0;
        sigj=0;
        for i=1:N
            muj=muj+phi(i,j)*x(:,i);
            mu(:,j)=muj/nj;
            sigj=sigj+phi(i,j)*(x(:,i)-mu(:,j))*(x(:,i)-mu(:,j))';
            sigma(:,:,j)=sigj/nj;
        end
        
        pi(j)=nj/N;
          
    end
    
    % compute log likelihood
    lnp=0;
    for i=1:N
        sum1=0;
        for j=1:K
            p(j)=pi(j)*mvnpdf(x(:,i),mu(:,j),sigma(:,:,j));
            sum1=sum1+pi(j)*mvnpdf(x(:,i),mu(:,j),sigma(:,:,j));    
        end
        lnp=lnp+log(sum1);
    end
    lnpx(t)=lnp;
    
    
end


%plot

figure(1)
plot(1:t,lnpx);
title('log likelihood when K=8')



figure(2)
%cpr=phi(:,1)-phi(:,2);
for i=1:250
    [maxp,ind]=max(phi(i,:));
    if ind==1
        scatter(x(1,i),x(2,i),'b');
        hold on
%         scatter(mu(1,ind),mu(2,ind),'bx');
%         hold on
    end
    if ind==2
        scatter(x(1,i),x(2,i),'r');
        hold on
%         scatter(mu(1,ind),mu(2,ind),'rx');
%         hold on
        
    end
    if ind==3
        scatter(x(1,i),x(2,i),'g');
        hold on
%         scatter(mu(1,ind),mu(2,ind),'gx');
%         hold on
        
    end
    if ind==4
        scatter(x(1,i),x(2,i),'c');
        hold on
%         scatter(mu(1,ind),mu(2,ind),'cx');
%         hold on
        
    end
    if ind==5
        scatter(x(1,i),x(2,i),'b<');
        hold on
%         scatter(mu(1,ind),mu(2,ind),'b>');
%         hold on
        
    end
    if ind==6
        scatter(x(1,i),x(2,i),'r<');
        hold on
%         scatter(mu(1,ind),mu(2,ind),'r>');
%         hold on
        
    end
    if ind==7
        scatter(x(1,i),x(2,i),'m<');
        hold on
%         scatter(mu(1,ind),mu(2,ind),'g>');
%         hold on
        
    end
    if ind==8
        scatter(x(1,i),x(2,i),'k');
        hold on
%         scatter(mu(1,ind),mu(2,ind),'mx');
%         hold on
        
    end
    if ind==9
        scatter(x(1,i),x(2,i),'m');
        hold on
%         scatter(mu(1,ind),mu(2,ind),'mx');
%         hold on
        
    end
    if ind==10
        scatter(x(1,i),x(2,i),'ys');
        hold on
%         scatter(mu(1,ind),mu(2,ind),'mx');
%         hold on
        
    end
%     if cpr(i)>=0
%         scatter(x(1,i),x(2,i),'b');
%         hold on
%     else
%         scatter(x(1,i),x(2,i),'r');
%         hold on
%     end
end
hold off
title('clustered result when K=10')
%legend('cluster1','cluster2','cluster3','cluster4','cluster5','cluster6','cluster7','cluster8');


