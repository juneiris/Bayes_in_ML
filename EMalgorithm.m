clear all;
data=load('/Users/xuchuwen/Downloads/mnist_mat.mat');
x_train=data.Xtrain;
y_train=data.ytrain;
x_test=data.Xtest;
y_test=data.ytest;
q=data.Q;
 
py0=zeros(length(x_test),1);
py1=zeros(length(x_test),1);

yresult=zeros(length(x_test),1);
y0r0=0;
y0r1=0;
y1r0=0;
y1r1=0;

T=100;
N=length(x_train(1,:));
w=zeros(15,1);
sigma=1.5;
lambda=1;


a=zeros(15,1);b=zeros(15,15);
sum1=0;sum2=0;
Eqt=zeros(1,N);
inp=zeros(1,T);
for t=1:T
    
    for i=1:N
        
        m=(x_train(:,i)')*w;

        if y_train(i)==1
            Eqt(i)=m+1.5*normpdf(-m/1.5)/(1-normcdf(-m/1.5));
        else
            Eqt(i)=m-1.5*normpdf(-m/1.5)/normcdf(-m/1.5);
        end
       
    end
    for i=1:N
       a=a+x_train(:,i)*Eqt(i);
       b=b+x_train(:,i)*x_train(:,i)';
       
    end 
    w=(eye(15)+b/2.25)\a/2.25;
    
    for i=1:N
       sum1 = sum1 + y_train(i)*log(normcdf(x_train(:,i)'*w/1.5));
       sum2 = sum2 + (1-y_train(i))*log(1-normcdf(x_train(:,i)'*w/1.5));
    end
    inp(t)=7.5*log(0.5/pi)-0.5*(w'*w)+sum1+sum2;
    
end
figure(1);
plot(1:T,inp)
title('lnp(y,w|X)')



%reconstruct w
figure(2)
w2=q*w;
pic=reshape(w2,28,28);
pic=mat2gray(pic);
imshow(pic) 
title('reconstruction of wt,t=100')






for j=1:length(x_test)
    py0(j)=1-normcdf(x_test(:,j)'*w/sigma,0,1);
    py1(j)=normcdf(x_test(:,j)'*w/sigma,0,1);
    ratio(j)=1+(py0(j)/py1(j));
    py1(j)=1/ratio(j);
    py0(j)=1-py1(j);
    
    diff(j)=abs(py0(j)-py1(j));
    
    if (py0(j)-py1(j))<0
        yresult(j)=1;
    else
        yresult(j)=0;
    end
end

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

% reconstruct pic that misclassified
figure(3)

x2(:,41)=q*x_test(:,41);
pic=reshape(x2(:,41),28,28);
pic=mat2gray(pic);
py0(41);
py1(41);
subplot(1,3,1)
imshow(pic) 
title('reconstruction of xtest(41)')


x2(:,65)=q*x_test(:,65);
pic=reshape(x2(:,65),28,28);
pic=mat2gray(pic);
py0(65);
py1(65);
subplot(1,3,2)
imshow(pic) 
title('reconstruction of xtest(65)')



x2(:,1039)=q*x_test(:,1039);
pic=reshape(x2(:,1039),28,28);
pic=mat2gray(pic);
py0(1039);
py1(1039);
subplot(1,3,3)
imshow(pic) 
title('reconstruction of xtest(1039)')

%reconstruct three most ambiguous
[minthree,index]=sort(diff);

x2(:,21)=q*x_test(:,21);
pic=reshape(x2(:,21),28,28);
pic=mat2gray(pic);
py0(21);
py1(21);
subplot(1,3,1)
imshow(pic) 
title('reconstruction of xtest(21)')


x2(:,318)=q*x_test(:,318);
pic=reshape(x2(:,318),28,28);
pic=mat2gray(pic);
py0(318);
py1(318);
subplot(1,3,2)
imshow(pic) 
title('reconstruction of xtest(318)')



x2(:,1447)=q*x_test(:,1447);
pic=reshape(x2(:,1447),28,28);
pic=mat2gray(pic);
py0(1447);
py1(1447);
subplot(1,3,3)
imshow(pic) 
title('reconstruction of xtest(1447)')












