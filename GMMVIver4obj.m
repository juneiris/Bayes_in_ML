clear all;
data=load('/Users/xuchuwen/Downloads/data.mat');
x=data.X;
d=size(x,1);
N=size(x,2);

a=d;
alpha=1;
c=10;

%set m,A to be the empirical mean of data
m=sum(x,2)/N;

A=cov(x');
B=(d/10)*A;

T=100;                                                                                                                                                  ;
K=2;

%sigma=eye(d,d);
%mj=eye(d,K)/10;
sigmaj=zeros(d,d,K);
aj=zeros(1,K);
Bj=zeros(d,d,K);
alphaj=zeros(1,K);

for j=1:K
    alphaj(j)=alpha;
    %mj(:,j)=10*[sin(pi*j/K);cos(pi*j/K)];
    mj(:,j)=j*(-1)^j*ones(size(x,1),1);
    %mj(:,j)=eye(d,1)/10*j;
    sigmaj(:,:,j)=eye(d,d);
    %sigmaj(:,:,j)=A;
    aj(j)=a;
    Bj(:,:,j)=B;
end

vof=zeros(1,T);
for t=1:T
    
    %update qci
    for i=1:N
        summ=0;
        t1=0;
        t2=0;
        t3=0;
        t4=0;
        for j=1:K
            t1j=0;
            for k=1:d
                t1j=t1j+psi((1-k+aj(j))/2);
            end
            t1(j)=t1j-log(det(Bj(:,:,j)));
%             t2(j)=(x(:,i)-mj(:,j))'*aj(j)/Bj(:,:,j)*(x(:,i)-mj(:,j));
%             t3(j)=trace(aj(j)*Bj(:,:,j)\sigmaj(:,:,j));
            t2(j)=(x(:,i)-mj(:,j))'*aj(j)/Bj(:,:,j)*(x(:,i)-mj(:,j));
            t3(j)=trace(aj(j)*Bj(:,:,j)\sigmaj(:,:,j));
            t4(j)=psi(alphaj(j))-psi(sum(alphaj));
            %phi(i,j)=exp(t1(j))*exp(t2(j))*exp(t3(j))*exp(t4(j));
            
            summ=summ+exp(0.5*t1(j)-0.5*t2(j)-0.5*t3(j)+t4(j));
        end
        for j=1:K
            phi(i,j)=exp(0.5*t1(j)-0.5*t2(j)-0.5*t3(j)+t4(j))/summ;
        end
        %phi(i,:)=phi(i,:)/summ;
       
        
    end


    
    %set nj,update q(pi), q(mu)..
    for j=1:K
        nj=sum(phi(:,j));
        
        alphaj(j)=alpha+nj;
        
        
        sigmaj(:,:,j)=eye(d)/(eye(d)/c+nj*aj(j)*eye(d)/Bj(:,:,j));
        %sigmaj(:,:,j)=(1/c)*eye(d,d)+nj*aj(j)*pinv(Bj(:,:,j));
        %sigmaj(:,:,j)=c^(-1)*eye(d,d)+nj*aj(j)./Bj(:,:,j);
        %sigmaj(:,:,j)=pinv(sigmaj(:,:,j));
        sum1=zeros(2,1);
        
        for i=1:N
            sum1=sum1+phi(i,j)*x(:,i);
        end
        mj(:,j)=sigmaj(:,:,j)*aj(j)/Bj(:,:,j)*sum1;
        sum2=zeros(d,d);
        for i=1:N
            sum2=sum2+phi(i,j)*((x(:,i)-mj(:,j))*(x(:,i)-mj(:,j))'+sigmaj(:,:,j));
        end
        aj(j)=a+nj;
        Bj(:,:,j)=B+sum2;
    end
    
    
    
    %compute variational objective function
    Epx=0;
    for i=1:N
        for j=1:K
            sum3=0;
            for k=1:d
               sum3=sum3+psi((aj(j)+1-k)/2);
            end
            EplnAj=d*(d-1)/4*log(pi)+d*log(2)-log(det(Bj(:,:,j)))+sum3;
            EmuA=(x(:,i)-mj(:,j))'*aj(j)/Bj(:,:,j)*(x(:,i)-mj(:,j))+trace(aj(j)*Bj(:,:,j)\sigmaj(:,:,j));
            Epxij=(-d/2)*log(2*pi)+0.5*phi(i,j)*EplnAj-0.5*phi(i,j)*EmuA;
            
            Epx=Epx+Epxij;
        end
    end
    
%     [mm,ci]=max(phi,[],2);
%     for j=1:K
%         Eppij(j)=alphaj(j)/sum(alphaj);
%     end
%     
%     Epc=0;
%     for i=1:N
%         Epc=Epc+ci(i)*Eppij(ci(i));
%     end
%     Epc=Epc/N;

    Epc=0;
    for i=1:N
        for j=1:K
            Epcij=phi(i,j)*(psi(alphaj(j))-psi(sum(alphaj)));
        
            Epc=Epc+Epcij;
        end
    end


    
    
    Elnppi=gammaln(K)-gammaln(1)*K;
    
    Elnpmu=0;
    for j=1:K
        Elnpmuj=(-d/2)*log(2*pi)-0.5*log(det(c*eye(d,d)))-0.5*(1/c)*trace((mj(:,j)*mj(:,j)'+sigmaj(:,:,j)));
        Elnpmu=Elnpmu+Elnpmuj;
    end
    
    ElnpA=0;
    for j=1:K
        sum4=0;
        for k=1:d
            sum4=sum4+psi((aj(j)+1-k)/2);
        end
        ElnAj=(d*(d-1)/4)*log(pi)+d*log(2)-log(det(Bj(:,:,j)))+sum4;
        EAj=aj(j)*pinv(Bj(:,:,j));
        
        sum5=0;
        for k=1:d
            sum5=sum5+gammaln((a+1-k)/2);
        end
        ElnpAj=(a-d-1)/2 * ElnAj -0.5*trace(B*EAj) -(a*d/2)*log(2)+(a/2)*log(det(B))-(d*(d-1)/4)*log(pi)-sum5;
        ElnpA=ElnpA+ElnpAj;
    end
    
    Ep=Epx+Epc+Elnppi+Elnpmu+ElnpA;
    
%     pij=sum(phi,1)/N;
%     Eqc=0;
%     for j=1:K
%             Eqcij=pij(j)*log(pij(j));
%             Eqc=Eqc+Eqcij;
%        
%     end
%     Eqc=-N*Eqc;
    Eqc=0;
    for i=1:N
        for j=1:K
            Eqcij=phi(i,j)*log(phi(i,j));
            
            Eqc=Eqc+Eqcij;
        end
    end
    
    Eqpi=gammaln(sum(alphaj));
    for j=1:K;
        Eqpij=(alphaj(j)-1)*(psi(alphaj(j))-psi(sum(alphaj)))-gammaln(alphaj(j));
        Eqpi=Eqpi+Eqpij;
    end
    
    Eqmu=0;
    for j=1:K
        Eqmuj=(-d/2)*log(2*pi)-0.5*log(det(sigmaj(:,:,j)))-d/2;
        Eqmu=Eqmu+Eqmuj;
    end
    
    
    
    EqA=0;
    for j=1:K
        sum6=0;
        for k=1:d
            sum6=sum6+psi((aj(j)+1-k)/2);
        end
        EqAj1=(d*(d-1)/4)*log(pi)+d*log(2)-log(det(Bj(:,:,j)))+sum6;
        EqAj2=aj(j)*pinv(Bj(:,:,j));
        sum7=0;
        for k=1:d
            sum7=sum7+gammaln((aj(j)+1-k)/2);
        end
        EqAj=(aj(j)-d-1)/2 * EqAj1 -0.5*trace(Bj(:,:,j)*EqAj2) -(aj(j)*d/2)*log(2)+(aj(j)/2)*log(det(Bj(:,:,j)))-(d*(d-1)/4)*log(pi)-sum7;
        
        
        EqA=EqA+EqAj;
    end
    
    
    Eq=Eqc+Eqpi+Eqmu+EqA;
    
    vof(t)=Ep-Eq;
    
    
end

figure(1)
plot(1:T,vof)
title('variational objective function')







figure(2)
for i=1:250   
    [maxp,ind]=max(phi(i,:));
    if ind==1
        scatter(x(1,i),x(2,i),'b');
        hold on
%         scatter(mj(1,ind),mj(2,ind),'bx');
%         hold on
    end
    if ind==2
        scatter(x(1,i),x(2,i),'r');
        hold on
%         scatter(mj(1,ind),mj(2,ind),'rx');
%         hold on
        
    end
    if ind==3
        scatter(x(1,i),x(2,i),'g');
        hold on
        
    end
    if ind==4
        scatter(x(1,i),x(2,i),'c');
        hold on
        
    end
    if ind==5
        scatter(x(1,i),x(2,i),'b<');
        hold on
        
    end
    if ind==6
        scatter(x(1,i),x(2,i),'r<');
        hold on
        
    end
    if ind==7
        scatter(x(1,i),x(2,i),'m<');
        hold on
        
    end
    if ind==8
        scatter(x(1,i),x(2,i),'k');
        hold on
        
    end
    if ind==9
        scatter(x(1,i),x(2,i),'m');
        hold on
        
    end
    if ind==10
        scatter(x(1,i),x(2,i),'ys');
        hold on
        
    end
end
hold off
title('clustered result when K=25')





