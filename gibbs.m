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
Kt=zeros(1,T);
%inidicator,initially all 1
C=ones(1,N);
%sample the first theta
lambda=zeros(d,d,100);
mu=zeros(d,100);
lambda(:,:,1)=wishrnd(pinv(B),a);
mu(:,1)=mvnrnd(m,pinv(c*lambda(:,:,1)));

csort=zeros(6,T);

hwait=waitbar(0,'Please wait>>>>>>>>');

for t=1:T
    str=['Running  ',num2str(t/T*100),'%']; waitbar(t/T,hwait,str);
    Kt(t)=K;
    for i=1:N
        for j=1:K
            idxj=find(C==j);
            nj=length(idxj);
            if nj~=0
                if C(i)==j
                    nj=nj-1;
                end
                
                phi(i,j)=mvnpdf(x(:,i),mu(:,j),inv(lambda(:,:,j)))*nj/250;
            else
                phi(i,j)=0;
                
            end
        end
        
        jnew=K+1;
        phi(i,jnew)=(alpha/(alpha+N-1))*(c/(pi*(1+c)))*det(B)*exp(gammaln(1.5)-gammaln(0.5))*(det(B+(c/(1+c))*(x(:,i)-m)*(x(:,i)-m)')^(-(a+1)/2));
        
        phi(i,:)=phi(i,:)/sum(phi(i,:));
        %[mm C(i)]=max(phi(i,:));
        C(i)=find(mnrnd(1,phi(i,:)));
        
        if C(i)==jnew
            K=K+1;
            
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
    
    j1=0;
    xs=zeros(d,1);
    lambda=zeros(d,d,100);
    mu=zeros(d,100);
    for j=1:K
        for i=1:N
            if C(i)==j
                xs=[xs x(:,i)];
            end
            
        end
        xs=xs(:,2:size(xs,2));
        s=size(xs,2);
        
        if isempty(xs)==0
            j1=j1+1;
            C(C==j)=j1;
            
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
            lambda(:,:,j1)=lambdajhat;
            mu(:,j1)=mujhat; 
        end
        
        
    end
    K=j1;
end
close(hwait);


figure(1)
plot(Kt);