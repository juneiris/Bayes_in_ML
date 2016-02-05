 data=load('/Users/xuchuwen/Documents/hw1_data_mat/mnist_mat.mat');
 x_train=data.Xtrain;
 y_train=data.ytrain;
 x_test=data.Xtest;
 y_test=data.ytest;
 q=data.Q;
 
y0r0=0;
y0r1=0;
y1r0=0;
y1r1=0;

 
N=length(x_train(1,:));
 
 
 %seperate xtrain into two classes according to y value
 l=1;
 k=1;
 
 for i=1:length(y_train)
     if y_train(i)==0
         classy0(l)=y_train(i);
         classx0(:,l)=x_train(:,i);
         l=l+1;
     else
         classy1(k)=y_train(i);
         classx1(:,k)=x_train(:,i);
         k=k+1;
     end
 end
 
N1=length(classy1);
N0=length(classy0);

%calculate parameters
xm0=mean(classx0,2);
xm1=mean(classx1,2);
% varx0=var(classx0);
% varx1=var(classx1);
for i=1:N1
    jian1(:,i)=(classx1(:,i)-xm1).^2;
end
for i=1:N0
    jian0(:,i)=(classx0(:,i)-xm0).^2;
end
varx1=sum(jian1,2);
varx0=sum(jian0,2);

mu1=(N1/(N1+1)).*xm1;
mu0=(N0/(N0+1)).*xm0;

k1=N1+1;
k0=N0+1;

alpha1=1+N1/2;
alpha0=1+N0/2;

beta1=1.+0.5.*varx1+0.5*(N1/(N1+1)).*xm1.^2;
beta0=1.+0.5.*varx0+0.5*(N0/(N0+1)).*xm0.^2;

% precision1=alpha1*k1./(beta1.*(k1+1));
% precision0=alpha0*k0./(beta0.*(k0+1));

py1=(1+N1)/(N+2);
py0=(1+N0)/(N+2);

% pxy1=tpdf(x_test,2*alpha1);
% pxy0=tpdf(x_test,2*alpha0);
gamma1=exp(gammaln(alpha1+0.5)-gammaln(alpha1));
gamma0=exp(gammaln(alpha0+0.5)-gammaln(alpha0));
% a1=(precision1./(2*alpha1)).^(0.5);
% a0=(precision0./(2*alpha0)).^(0.5);'
a1=(k1./(2.*beta1.*(k1+1))).^0.5;
a0=(k0./(2.*beta0.*(k0+1))).^0.5;



for j=1:length(x_test)
    
    b1(:,j)=(1+((x_test(:,j)-mu1).^2).*a1.*a1).^(-alpha1-0.5);
    b0(:,j)=(1+((x_test(:,j)-mu0).^2).*a0.*a0).^(-alpha0-0.5);
    
    
    pxy1(:,j)=gamma1*pi^(-0.5)*a1.*b1(:,j);
    pxy0(:,j)=gamma0*pi^(-0.5)*a0.*b0(:,j);
end
pxy1_all=ones(1,length(x_test));
pxy0_all=ones(1,length(x_test));

for i=1:length(x_test)
    for j=1:15
        pxy1_all(:,i)=pxy1_all(:,i).*pxy1(j,i);
        pxy0_all(:,i)=pxy0_all(:,i).*pxy0(j,i);
    end
end
% pxy1_all=mean(pxy1,1);
% pxy0_all=mean(pxy0,1);


%get result
presult1=pxy1_all*py1;
presult0=pxy0_all*py0;


for i=1:length(presult1)
    ratio(i)=1+(presult0(i)/presult1(i));
    presult1(i)=1/ratio(i);
    presult0(i)=1-presult1(i);
   
end


for i=1:length(presult1)
    if (presult0(i)-presult1(i))<0
        yresult(i)=1;
    else
        yresult(i)=0;
    end
end

%evaluation

for i=1:length(y_test)
    if (y_test(i)==0)&&(yresult(i)==0)
        y0r0=y0r0+1;
    end
    if (y_test(i)==0)&&(yresult(i)==1)
        y0r1=y0r1+1;
    end
    if (y_test(i)==1)&&(yresult(i)==0)
        y1r0=y1r0+1;
    end
    if (y_test(i)==1)&&(yresult(i)==1)
        y1r1=y1r1+1;
    end
                
        
    
    
end
y0r0
y0r1
y1r0
y1r1


% reconstruct pic that misclassified
figure(1)

x2(:,56)=q*x_test(:,56);
pic=reshape(x2(:,56),28,28);
pic=mat2gray(pic);
presult0(56);
presult1(56);
subplot(1,3,1)
imshow(pic) 
title('reconstruction of xtest(56)')


x2(:,184)=q*x_test(:,184);
pic=reshape(x2(:,184),28,28);
pic=mat2gray(pic);
presult0(184);
presult1(184);
subplot(1,3,2)
imshow(pic) 
title('reconstruction of xtest(184)')


x2(:,1317)=q*x_test(:,1317);
pic=reshape(x2(:,1317),28,28);
pic=mat2gray(pic);
presult0(1317);
presult1(1317);
subplot(1,3,3)
imshow(pic) 
title('reconstruction of xtest(1317)')

%reconstruct pic that are most ambiguous


    
x2(:,358)=q*x_test(:,358);
pic=reshape(x2(:,358),28,28);
pic=mat2gray(pic);
presult0(358);
presult1(358);
figure(2)
subplot(1,3,1)
imshow(pic) 
title('reconstruction of xtest(358)');

x2(:,823)=q*x_test(:,823);
pic=reshape(x2(:,823),28,28);
pic=mat2gray(pic);
presult0(823);
presult1(823);
subplot(1,3,2)
imshow(pic) 
title('reconstruction of xtest(823)');

% 
x2(:,1131)=q*x_test(:,1131);
pic=reshape(x2(:,1131),28,28);
pic=mat2gray(pic);
presult0(1131);
presult1(1131);
subplot(1,3,3)
imshow(pic)
title('reconstruction of xtest(1131)');
 






 
 
 
 
 
 
 
 