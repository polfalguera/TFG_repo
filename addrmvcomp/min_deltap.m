function [deltap,deltapprima,StoR,StoE,EtoS,RtoS]=min_deltap(X,t,xm1,C,alpha_ini,b_ini,S,R,E,invQ,p,target)

epsilon=0.00000000000001;

precision=0.00000000000000000001;

N=log(precision)/log(0.5); %number of iterations: logarithm base 0.5 of precision
N=N*(1-p);

if target == -1
    delta_pMin=target-p;
    delta_pMax=0;
else
    delta_pMin=0;
    delta_pMax=target-p;
end

alpha=alpha_ini;
malpha=alpha;
b=b_ini;
w=((alpha.*t)'*[X sqrt(p)*xm1])';
my=[X sqrt(p)*xm1]*w+b;

deltap=delta_pMin;
deltapprima=delta_pMax;
uplus=[0; (t(S,:).*xm1(S,:))]; 
for i=1:N,
    cambios=0;
    delta_p=(delta_pMin+delta_pMax)/2;
    invQU=delta_p*(invQ-(delta_p*(invQ*uplus)*(uplus'*invQ))/(1+delta_p*uplus'*invQ*uplus)); 
    suma_E=xm1(E)'*t(E); 
    V=[0; C*suma_E*(xm1(S).*t(S))]; 
    
    incrementos= -invQU*(uplus*(uplus'*[0;alpha_ini(S)])+V); 

    b=b_ini+incrementos(1);
    alpha(S)=alpha_ini(S)+incrementos(2:end);


    w=((alpha.*t)'*[X sqrt(p+delta_p)*xm1])'; 
    y=[X sqrt(p+delta_p)*xm1]*w+b; 

    if (sum((y(R).*t(R))<=1-epsilon)+sum((y(E).*t(E))>=1+epsilon)>0) 
        cambios=1;
    else
        if (sum(alpha(S)>C+epsilon)+sum(alpha(S)<0-epsilon)>0)
            cambios=1;
        end
    end

    if (cambios==0)
        delta_pMin=delta_p;
        deltap=delta_p;
    else
        delta_pMax=delta_p;
        deltapprima=delta_p;
        my=y;
        malpha=alpha;
    end
end

my=my.*t;
RtoS=~ones(length(t),1); 
RtoS(R)=my(R)<=1-epsilon;
EtoS=~ones(length(t),1);
EtoS(E)=my(E)>=1+epsilon;
StoR=~ones(length(t),1);
StoR(S)=my(S)>(1+epsilon);
StoR(S)=malpha(S)<0-epsilon;

StoE=~ones(length(t),1); 
StoE(S)=my(S)<(1-epsilon);
StoE(S)=malpha(S)>C+epsilon;
CAMBIOS=sum([StoR; StoE; EtoS; RtoS]);
if (CAMBIOS==0)
   deltap=deltapprima;
end
