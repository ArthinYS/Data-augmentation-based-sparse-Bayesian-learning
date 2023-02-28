%% We apply MCMC method to identify the Langevin equation in this code
%% Langevin equation dx=-bx*dt+a*dW

clc
clear
close all

%%  Euler-Maruyama 
randn('state',100)

n=5000;%% Dt  statisfies Dt=ndt where n is a integer.
dt=0.001;x(1)=0;a=0.5;
 for i=1:n/dt
     x(i+1)=x(i)-x(i)*dt+a*sqrt(dt)*randn;
 end
 Dt=0.3; %n=2000;%% Dt  statisfies Dt=ndt where n is a integer.
 Xi(1)=1;
 Tr(1)=0;
 npp=5000;
 for i=1:npp
     s=randperm(3,1);
     if s==1
         Xi(i+1)=Xi(i)+200;
         Tr(i+1)=Tr(i)+0.2;
     elseif s==2
         Xi(i+1)=Xi(i)+300;
         Tr(i+1)=Tr(i)+0.3;
     else 
         Xi(i+1)=Xi(i)+400;
         Tr(i+1)=Tr(i)+0.4;
     end
 end
 
 for i=1:npp+1
     X(i)=x(Xi(i));
 end
         
 %% Plot
 

 

 %% MCMC 
 hp=1;dtt=0.1; T1=0:Dt:n; %X=zeros(1,n/dtt+1,40001);
% hp=1;dtt=Dt/(hp+1); T1=0:Dt:n;
 %aa=n/dtt;
% X=zeros(1,15001,30001);
%Phi={'1','x','x^2','x^3','x^4','x^5'};
Phi={'1','x','x^2','exp(x)','exp(2*x)','exp(3*x)'};
%Phi={'1','x','x^2','x^3','x^4','x^5','x^6','x^7'};
 fx_name={'1','u','u^2','exp(u)','exp(2*u)','exp(3*u)'};

 tic
 [sies,thes,X,X_P,X2_P,th_P,ga_P,si_P,XAR,XAR_rate,Rv]=GibbsSampler_NM3(X,Tr,dtt,Phi);
 toc
threshold = 0.01;
y_name = 'b(x(t))';
fprintf('%s = ', y_name);
W2=th_P;
for i = 1:size(W2,2)
    if abs(W2(i))<threshold
       ;
    else
       if W2(i)<0
           fprintf('%.4f%s', W2(i),fx_name{i});        
       else
           fprintf('+');
           fprintf('%.4f%s', W2(i),fx_name{i});
        end
    end
end
fprintf('\n')
 















 
%%
 
 function [sies,thes,X,X_P,X2_P,th_P,ga_P,si_P,XAR,XAR_rate,Rv]=GibbsSampler_NM3(Z,T1,dt,Phi)  %x column vector    t row vector
        %% Illustration
        %%% Z: p*(n+1)  p denotes the dimension of SDEs and (n+1) is the
        %%%  total number of measurements
        %%% T1: 1*(n+1) 
        %%% dt: The finer time step
        %%% hp: The number of hidden states
        %%% Phi: The selected basis functions 
    
        

%% initialization
        Phi=[];
        a=1e-4;b=1e-4;MI=40000;ep=0.2;% ep denotes the step size
        T=[T1(1):dt:T1(end)];
        n = length(T1)-1;N = length(T)-1;                                   
        t_ind = ismembertol(T,T1); % For example: t_ind=[1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1]
        t_ind1 = find(t_ind); % For example: [1 5 9 13]
        t_ind2 = find(t_ind==0);
        p=size(Z,1);
        XX =interp1(T1',Z',T','linear')';%p*(N+1)
        X2(:,:,1)=XX(:,t_ind2);% p*(hp*n)
        X(:,t_ind1,1)=Z;
        X(:,t_ind2,1)=X2(:,:,1);
        %OX=XX(:,2:end);UX=XX(:,1:end-1);dX=OX-UX;% p*N
        OX=X(:,2:end,1);UX=X(:,1:end-1,1);dX=OX-UX;% p*N
        Phi=PhiBF(UX); % q*N
        q=size(Phi,1);% q denotes the number of basis functions; 
       % si(:,1)=ones(p,1);ga(:,:,1)=0.1*ones(p,q);th(:,:,1)=randn(p,q);
         si(:,1)=ones(p,1);ga(:,:,1)=0.1*ones(p,q);th(:,:,1)=ones(p,q);
        XAR=0;
        %% symbols which  we don't need to calculate in each iteration 
       % cof=1:hp;cof=cof/(hp+1); cof=repmat(cof,1,n); cof=diag(cof); %np*np
        %for i=1:hp
        %  for j=i:hp
        %    P(i,j)=i*(hp+1-j)/(hp+1);
        %  end
       % end
       % P=P+P'-diag(diag(P));
       % P
       % L=chol(P)';
       % mc=Z(:,2:end)-Z(:,1:end-1);
       % mcp=kron(mc,ones(1,hp))*cof; 
       % m2=kron(Z(:,1:end-1),ones(1,hp))+mcp;
       %% Iterations
         for i=1:MI
             if mod(i,500)==0;
                 fprintf('Iteration: %d \n',i)
             end            
     %% Sample X(T2)
     inal=[];
         for ii=1:p           
               for jj=1:n
                   hh1=T1(jj+1)-T1(jj);
                   hh2=round(hh1/dt);
                   hp=hh2-1;
                   hh3=1:hp;
                   inali=size(inal,2)+hh3;
                   inal=[inal,inali];
                  % e1=m2(ii,(jj-1)*hp+1:jj*hp);
                   e1=Z(jj)+[1:hp]./hh2*(Z(jj+1)-Z(jj));
                   e2=X2(ii,inali,i)-e1;
                   if hp==1
                       Kii=[2];
                       P=inv(Kii);
                   elseif hp==2
                       kii=[2,-1;-1,2];
                       P=inv(kii);
                   else
                       kii=[2,-1,0;-1,2,-1;0,-1,2];
                       P=inv(kii);
                   end
                     L=chol(P)';
                     
                     X2(ii,inali,i+1) = (e1'+sqrt(1-ep^2)*e2'+ep*sqrt(si(ii,i)*dt)*L*randn(hp,1))';
               end
         end
         
     
      %% Formulate acceptance probability.
      
      Xk(:,t_ind1,i)=Z;Xk(:,t_ind2,i)=X2(:,:,i);
      OX=Xk(:,2:end,i);UX=Xk(:,1:end-1,i);dxk=OX-UX;
      Phik=PhiBF(UX);
 
      Xp(:,t_ind1,i)=Z;Xp(:,t_ind2,i)=X2(:,:,i+1);
      OX=Xp(:,2:end,i);UX=Xp(:,1:end-1,i);dxp=OX-UX; 
      Phip=PhiBF(UX);     
      R=1;
      
      for jj=1:p
          K1=diag((1./ga(jj,:,i)))*si(jj,i)+dt*Phip*Phip';
          [U1,S1,V1]=svd(K1);
          K2=diag((1./ga(jj,:,i)))*si(jj,i)+dt*Phik*Phik';
          [U2,S2,V2]=svd(K2);
          r1=(det(S2)/det(S1))^(0.5);
          
          KK1=dxp(jj,:)*Phip'*U1;
          KK2=dxk(jj,:)*Phik'*U2;
          
          ga(jj,:,i);
          cc1= KK1*diag(1./diag(S1))*KK1';
          cc2=KK2*diag(1./diag(S2))*KK2';
   
          
          r2=exp(1/(2*si(jj,i))*(KK1*diag(1./diag(S1))*KK1'-KK2*diag(1./diag(S2))*KK2'));
          % Another choice
% % % % % %           KK1=dxp(jj,:)*Phip'*inv(K1)*Phip*dxp(jj,:)';
% % % % % %           KK2=dxk(jj,:)*Phik'*inv(K2)*Phik*dxk(jj,:)';
% % % % % %           r2=exp(1/(2*si(jj,i))*(KK1-KK2))
          R=R*r1*r2;  
      end
      Rv(i)=R;

           acc=rand;
           
        if acc<= min(1,R)
            X2(:,:,i+1)=X2(:,:,i+1);
            XAR=XAR+1;
          %  fprintf(' this is %d-th iteration, and we accept its sampled trajectory\n',i)
        else
            X2(:,:,i+1)= X2(:,:,i);    
          %  fprintf(' this is %d-th iteration, and we refuse its sampled trajectory\n',i)
        end
       
% % % % % % % % % %% Formulate acceptance probability
% % % % % % % % %          for jj=1:p
% % % % % % % % %              r1=det(diag(si(jj,i)*(1./ga(jj,:,i)))+dt*Phip*Phip')^(-0.5)*exp(1/(2*si(jj,i))*dxp(jj,:)*Phip'*inv(si(jj,i)*diag(1./ga(jj,:,i))+dt*Phip*Phip')*Phip*dxp(jj,:)');
% % % % % % % % %              r2=det(diag(si(jj,i)*(1./ga(jj,:,i)))+dt*Phik*Phik')^(-0.5)*exp(1/(2*si(jj,i))*dxk(jj,:)*Phik'*inv(si(jj,i)*diag(1./ga(jj,:,i))+dt*Phik*Phik')*Phik*dxk(jj,:)');
% % % % % % % % %              R=R*r1/r2
% % % % % % % % %          end
% % % % % % % % %       

         X(:,t_ind1,i+1)=Z;X(:,t_ind2,i+1)=X2(:,:,i+1);
   
         OX=X(:,2:end,i+1);UX=X(:,1:end-1,i+1);
         dx=OX-UX; 
         Phi=PhiBF(UX);
     
         
      %% Sample theta
        for  jj=1:p
             Ar=pinv(dt*Phi*Phi'/si(jj,i)+diag(1./ga(jj,:,i)));%\eye(q);
             OHr=1/si(jj,i)*dx(jj,:)*Phi'*Ar;
             Lth=chol(Ar)';
             th(jj,:,i+1)=(OHr'+Lth*randn(q,1))';
        end
        
         
                   
      %% sample gamma     
                      for ii=1:p
                          for jj=1:q
                              
                              ga(ii,jj,i+1)=sampig(a+0.5,b+0.5*th(ii,jj,i+1)^2);
                          end
                         
                      end
       %% sample sigma
       for jj=1:p
                   sishape=a+N/2;
                   err2(jj)=sum((dx(jj,:)-(th(jj,:,i+1)*Phi*dt)).^2);
                   siscale=b+err2(jj)/(2*dt);
                   si(jj,i+1)=sampig(sishape,siscale);
       end
    
         end
      sies=si;
      thes=th;
  
      %%  Infer mode of each random variable
   X2_P=zeros(p,size(t_ind2,2));
   X_P=zeros(p,N+1); si_P=zeros(p,1);ga_P=zeros(p,q);th_P=zeros(p,q);
            for jj=MI/2:MI
                X2_P=X2_P+X2(:,:,jj);
                X_P=X_P+X(:,:,jj);
                th_P=th_P+th(:,:,jj);
                ga_P=ga_P+ga(:,:,jj);
                si_P=si_P+si(:,jj);
            end
            
            IMP=MI/2+1;
            X2_P=X2_P/IMP;
            X_P=X_P/IMP;
            th_P=th_P/IMP;
            ga_P=ga_P/IMP;
            si_P=si_P/IMP; 
            XAR_rate = XAR/MI;
end
         
%%  Sample inverse gamma distribution
   function IG=sampig(x,y)
                  IGG=gamrnd(x,1/y);
                  IG=1/IGG;
   end
%% Formulate library matrix
function phi=PhiBF(X)
        ss=size(X,2);     
       %phi=[X;X.^2;X.^3;X.^4];  
      %  phi=[X;X.^2];
      phi=[ones(1,ss);X;X.^2;exp(X);exp(2*X);exp(3*X)];
%phi=[ones(1,ss);X;X.^2;X.^3;X.^4;X.^5];
 %phi=[ones(1,ss);X;X.^2;X.^3;X.^4;X.^5;X.^6;X.^7];
end




















%%
 function [X,gamma_ind,gamma_est,count,gamm,lambda] = MSBL1(Phi, Y, lambda, Learn_Lambda,varargin)
% Sparse Bayesian Learning for Mulitple Measurement Vector (MMV) problems. 
% *** The version is suitable for noisy cases ***
% It can also be used for single measurement vector problem without any modification.
%
% Command Format:
% [X,gamma_ind,gamma_est,count] ...
%     = MSBL(Phi,Y,lambda,Learn_Lambda,'prune_gamma',1e-4,'max_iters',500,'epsilon',1e-8,'print',0);
% [X,gamma_ind,gamma_est,count] = MSBL(Phi,Y, lambda, Learn_Lambda, 'prune_gamma', 1e-4);
% [X,gamma_ind,gamma_est,count] = MSBL(Phi,Y, lambda, Learn_Lambda);
%
% ===== INPUTS =====
%   Phi         : N X M dictionary matrix
%
%   Y           : N X L measurement matrix
%
%   lambda      : Regularization parameter. Sometimes you can set it being the
%                 noise variance value, which leads to sub-optimal
%                 performance. The optimal value is generally slightly larger than the
%                 noise variance vlaue. You need cross-validation methods or
%                 other methods to find it. 
%
%  Learn_Lambda : If Learn_Lambda = 1, use the lambda as initial value and learn the optimal lambda 
%                 using its lambda learning rule. But note the
%                 learning rule is not robust when SNR <= 20 dB. 
%                    If Learn_Lambda = 0, not use the lambda learning rule, but instead, use the 
%                 input lambda as the final value.
%
%  'PRUNE_GAMMA' : Threshold for prunning small hyperparameters gamma_i.
%                  In noisy cases, you can set MIN_GAMMA = 1e-3 or 1e-4.
%                  In strong noisy cases (e.g. SNR < 5 dB), set MIN_GAMMA = 1e-2 for better 
%                  performance.
%                   [ Default value: MIN_GAMMA = 1e-4 ]
%
%  'MAX_ITERS'   : Maximum number of iterations.
%                    [ Default value: MAX_ITERS = 2000 ]
%
%  'EPSILON'     : Threshold to stop the whole algorithm. 
%                    [ Default value: EPSILON = 1e-8   ]
%
%  'PRINT'       : Display flag. If = 1: show output; If = 0: supress output
%                    [ Default value: PRINT = 0        ]
%
% ===== OUTPUTS =====
%   X          : the estimated solution matrix, or called source matrix (size: M X L)
%   gamma_ind  : indexes of nonzero gamma_i
%   gamma_est  : final value of the M X 1 vector of hyperparameter values
%   count      : number of iterations used
%
%
% *** Reference ***
% [1] David P. Wipf, Bhaskar D. Rao, An Empirical Bayesian Strategy for Solving
%     the Simultaneous Sparse Approximation Problem, IEEE Trans. Signal
%     Processing, Vol.55, No.7, 2007.
%
% *** Author ***
%   Zhilin Zhang (z4zhang@ucsd.edu) 
%   (Modified based on David Wipf's original code such that the code is suitable for noisy cases)
%
% *** Version ***
%   1.1 (02/12/2011)
%
% *** See Also ***
%   TSBL        TMSBL
%
  


% Dimension of the Problem
[N M] = size(Phi); 
[N L] = size(Y);  

% Default Control Parameters 
%PRUNE_GAMMA = 1e-4;
PRUNE_GAMMA = 1e-4;% threshold for prunning small hyperparameters gamma_i
EPSILON     = 1e-8;       % threshold for stopping iteration. 
MAX_ITERS   = 700;       % maximum iterations
PRINT       = 0;          % don't show progress information


if(mod(length(varargin),2)==1)
    error('Optional parameters should always go by pairs\n');
else
    for i=1:2:(length(varargin)-1)
        switch lower(varargin{i})
            case 'prune_gamma'
                PRUNE_GAMMA = varargin{i+1}; 
            case 'epsilon'   
                EPSILON = varargin{i+1}; 
            case 'print'    
                PRINT = varargin{i+1}; 
            case 'max_iters'
                MAX_ITERS = varargin{i+1};  
            otherwise
                error(['Unrecognized parameter: ''' varargin{i} '''']);
        end
    end
end

if (PRINT) fprintf('\nRunning MSBL ...\n'); end


% Initializations 

gamma = 100*rand(M,1)+0.1; 
%gamma=kkk*rand(M,1);
keep_list = [1:M]';% M: dimension
m = length(keep_list);
mu = zeros(M,L);
count = 0;                        % iteration count


% *** Learning loop ***
while (1)

    % *** Prune weights as their hyperparameters go to zero ***
    if (min(gamma) < PRUNE_GAMMA )
        index = find(gamma > PRUNE_GAMMA);
        gamma = gamma(index);  % use all the elements larger than MIN_GAMMA to form new 'gamma'
        Phi = Phi(:,index);    % corresponding columns in Phi
        keep_list = keep_list(index);
        m = length(gamma);
    end;
   
    if count == 1
        gamm = gamma;
    end

    mu_old =mu;
    Gamma = diag(gamma);
    G = diag(sqrt(gamma));
        
    % ****** estimate the solution matrix *****
    [U,S,V] = svd(Phi*G,'econ');
   
    [d1,d2] = size(S);
    if (d1 > 1)     diag_S = diag(S);
    else            diag_S = S(1);    
  end;
       
    Xi = G * V * diag((diag_S./(diag_S.^2 + lambda + 1e-16))) * U'; % why add le-16
    mu = Xi * Y;
    
    % *** Update hyperparameters, i.e. Eq(18) in the reference ***
    gamma_old = gamma;
    mu2_bar = sum(abs(mu).^2,2)/L;

    Sigma_w_diag = real( gamma - (sum(Xi'.*(Phi*Gamma)))');
    gamma = mu2_bar + Sigma_w_diag;

    % ***** the lambda learning rule *****
    % You can use it to estimate the lambda when SNR >= 20 dB. But when SNR < 20 dB, 
    % you'd better use other methods to estimate the lambda, since it is not robust 
    % in strongly noisy cases (in simulations, you can feed it with the
    % true noise variance, which can lead to a near-optimal performance)
    if Learn_Lambda == 1
        lambda = (norm(Y - Phi * mu,'fro')^2/L)/(N-m + sum(Sigma_w_diag./gamma_old));   
    end;
    
    
    % *** Check stopping conditions, etc. ***
    count = count + 1;
    if (PRINT) disp(['iters: ',num2str(count),'   num coeffs: ',num2str(m), ...
            '   gamma change: ',num2str(max(abs(gamma - gamma_old)))]); end;
    if (count >= MAX_ITERS) break;  end;

    if (size(mu) == size(mu_old))
        dmu = max(max(abs(mu_old - mu)));
        if (dmu < EPSILON)  break;  end;
    end;

end;


% Expand hyperparameters 
gamma_ind = sort(keep_list);  % the row which was not deleted
gamma_est = zeros(M,1);       % 
gamma_est(keep_list,1) = gamma; % the value of the gamma which was not deleted 

% expand the final solution
X = zeros(M,L);
X(keep_list,:) = mu; 

if (PRINT) fprintf('\nFinish running ...\n'); end
return;
 end

