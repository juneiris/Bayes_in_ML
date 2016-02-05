clear all;
data=load('/Users/xuchuwen/Downloads/data.mat');
x=data.X;
d=size(x,1);
N=size(x,2);

%et parameteres
m=mean(x,2);
A=cov(x');

c=1/10;
a=d;
B=c*d*A;

alpha=1;

T=500;
%cluster num, initially 1 
K=1;
%inidicator,initially all 1
C=ones(1,N);
%sample the first theta
lambda=wishrnd(inv(B),a);
mu=(mvnrnd(m,inv(c*lambda)))';



%initial nj
nj=N;

Kt=zeros(1,T);
first=zeros(1,T);
second=zeros(1,T);
third=zeros(1,T);
fourth=zeros(1,T);
fifth=zeros(1,T);
sixth=zeros(1,T);


for t=1:T
    
    Kt(t)=size(nj(nj~=0),2);
    nsort=sort(nj(nj~=0),'descend');
    if size(nsort)<6
        az=zeros(1,6-size(nsort));
        nsort=[nsort az];
    end
    first(t)=nsort(1);
    second(t)=nsort(2);
    third(t)=nsort(3);
    fourth(t)=nsort(4);
    fifth(t)=nsort(5);
    sixth(t)=nsort(6);
    
    
    for i=1:N
        for j=1:K
            
            idxj=find(C==j);
            nj(j)=length(idxj);
            if isempty(idxj)==0
                if C(i)==j
                    nj(j)=nj(j)-1;
                    %nj=nj-1;
                end
                
                phi(i,j)=mvnpdf(x(:,i),mu(:,j),inv(lambda(:,:,j)))*nj(j)/(alpha+N-1);
            else
                phi(i,j)=0;
            end
            
        end
        %compute probability that xi not belongs to existed clusteres
        temp1=(c/(pi*(1+c)))^(d/2);
        temp2=det(B+(c/(1+c))*(x(:,i)-m)*(x(:,i)-m)');
        temp2=temp2^(-(a+1)/2);
        temp3=det(B)^(-a/2);
        temp4=exp(gammaln(1.5)-gammaln(0.5));
        pxinewj=temp1*(temp2/temp3)*temp4;
        phinewj=alpha*pxinewj/(alpha+N-1);
        
        %normalize phi and sample C
        j1=K+1;
        phi(i,j1)=phinewj;
        phi(i,:)=phi(i,:)/sum(phi(i,:));
        %C(i)=discretesample(phi(i,:),1);
        C(i)=find(mnrnd(1,phi(i,:)));
        
        if C(i)==j1
            K=K+1;
            %generate theta for the new cluster
            mnew=m*c/(c+1)+x(:,i)/(c+1);
            cnew=c+1;
            anew=a+1;
            Bnew=B+(1/(a+1))*(x(:,i)-m)*(x(:,i)-m)';
            lambdanew=wishrnd(inv(Bnew),anew);
            munew=mvnrnd(mnew,inv(cnew*lambdanew));
            lambda(:,:,K)=lambdanew;
            mu(:,K)=munew;
            
        end
      
        
    end
    
    
    
    %update theta for each K
    
    %divide data into clusters
    xs=zeros(d,1);
    for j=1:K
        for i=1:N
            if C(i)==j
                xs=[xs x(:,i)];
            end
            
        end
        xs=xs(:,2:size(xs,2));
        s=size(xs,2);
        
        if isempty(xs)==0
            mjhat=m*c/(s+c)+sum(xs,2)/(s+c);
            cjhat=s+c;
            ajhat=a+s;
            Bjhat=B+(s/(a*s+1))*(mean(xs,2)-m)*(mean(xs,2)-m)';
            for i=1:s
                Bjhat=Bjhat+(xs(:,i)-mean(xs,2))*(xs(:,i)-mean(xs,2))';
            end
            %Bjhat=B+sum1+(s/(a*s+1))*(mean(xs,2)-m)*(mean(xs,2)-m)';
            lambdajhat=wishrnd(inv(Bjhat),ajhat);
            mujhat=mvnrnd(mjhat,inv(cjhat*lambdajhat));
            lambda(:,:,j)=lambdajhat;
            mu(:,j)=mujhat; 
            
        end
        
    end
end



figure(1)
plot(first,'b');
hold on
plot(second,'r');
hold on
plot(third,'g');
hold on
plot(fourth,'m');
hold on
plot(fifth,'k');
hold on
plot(sixth,'c');
hold off
title('numbers of data in six most propable clusters')


figure(2)
plot(Kt)
title('total number of clusters')



