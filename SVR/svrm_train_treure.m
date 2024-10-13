function [svmState, continued] = svrm_train_treure(svmState, x, y, xnew, ynew, C, epsilon);  
% Add epsilon - MM 14/11/01
% function [a, b, g, inds, inde, indw] = svcm_train(x, y, C);
%        support vector classification machine
%        incremental learning, and leave-one-out cross-validation
%        soft margin
%        uses "kernel.m"
%
%        x: independent variable, (L,N) with L: number of points; N: dimension
%        y: dependent variable, (L,1) containing class labels (-1 or +1)
%        C: soft-margin regularization constant
%
%        a: alpha coefficients 
%        b: offset coefficient
%        g: derivatives (adding one yields margins for each point)
%        inds: indices of support vectors in S
%        indss: indices of support vectors in S*
%        inde: indices of error vectors in E
%        indes: indices of error star vectors 
%%%%%%%%%% version 0.9; last revised 02/12/2001; send comments to gert@jhu.edu %%%%%%%%%%

global verbose
global debug
a = svmState.a;
b = svmState.b;
g = svmState.g;
inds = svmState.inds; 
indss = svmState.indss;
inde = svmState.inde;
indes = svmState.indes;
support = svmState.support;
ing = svmState.ing;
ings = svmState.ings;
indo = svmState.indo;
R = svmState.R;
Qs = svmState.Qs;
Qc = svmState.Qc;
processed = svmState.processed;
eps = svmState.eps;
eps2 = svmState.eps2;
tol = svmState.tol;

[L,N] = size(x);

for i=1:length(xnew)
   indnew(i)= find(x==xnew(i));
end

%g(indnew)= -(ynew - y(indnew));
%yold=y(indnew);
%y(indnew)= ynew;

%fprintf('Support vector soft-margin regression with incremental learning\n')
%fprintf('  %g training points\n', L)
%fprintf('  %g dimensions\n\n', N)
[L,N] = size(x);

ls = length(inds)+length(indss);  % number of support vectors;
le = length(inde);                % number of error vectors;
les = length(indes);              % number of error vectors; - MM 14/11/01
la = ls+le+les;                   % both                     - MM 14/11/01
lo = length(indo);                % number of other vectors;

kernelcount = 0;
iter = 0;                         % iteration count

indc = 0;                             % candidate vector
indso = 0;                            % a recycled support vector; used as buffer

%free = a(indnew)<-tol|a(indnew)>tol; % free, candidate support or error vector
free=1;
left= indnew(free);
%if ~any(free)
  %free = g(indo)<0|2*epsilon-g(indo)<0; % free, candidate support or error vector
  %left = indo(free);                    % candidates left
%  for k=1:length(indnew),
%   i(k) = find(indo==indnew(k));
%  end
%  indo(i) = [];  % remove indc from indo
%  lo = lo-1;
%  indnew=[];
%end

continued = any(left);

while continued                   % check for remaining free points or leave-one-outs to process
    
    % select candidate indc
    indc_prev = indc;
    
    indc = left(length(left));        % take top of the stack, "last-in, first-out
    
    if indc_prev>0
%        processed(1:indc_prev) = 1;   % record last and all preceding point
         processed(1:indc-1)=1;
    end       
       
    if verbose==1
       if indc~=indc_prev
            fprintf('\n%g',indc)
       else
            fprintf('o')
       end
    end   
    
%    if ~any(find(indc==indo))
%    if indc==5
%       fprintf('')
%    end

    % get Qc, row of hessian corresponding to indc (needed for gamma)
    xc = x(indc,:);
    Qc = kernelsvr(xc,x);            %  En classificació  Qc = (yc*y').*kernel(xc,x);
    Qc(indc) = Qc(indc)+eps2;
    
    % prepare to increment/decrement z = a(indc)' or y(indc)*b, subject to constraints.
    % move z up when adding indc ((re-)training), down when removing indc (leave-one-out or g>0)
    upc = (a(indc)<0);
    
    if any(inde)
        if any(find(indc==inde))
             upc=0; 
        end
    end

    if any(indes)
       if any(find(indc==indes))
          upc=1;
       end 
    end
 
    if any(inds)      
       if any(find(indc==inds))
          upc=0;
       end 
    end
   
    if any(indss)  
       if any(find(indc==indss))
         upc=1;  
       end
    end
   
    polc = 2*upc-1;                % polarity of increment in z
    beta = -R*Qs(:,indc);          % change in [b;a(inds)] per change in a(indc)
    if ls>0
        % move z = a(indc)'
        gamma = Qc'+Qs'*beta;      % change in g(:) per change in z = a(indc)'
        z0 = a(indc);              % initial z value
        zlim = (0-z0);             % constraint on a(indc)
    else % ls==0
        % move z = b and keep a(indc) constant; there is no a(:) free to move in inds!
        gamma = Qs';               % change in g(:) per change in z = b
        z0 = b;                    % initial z value
        zlim = (0-z0);           % no constraint on b
    end
    
    gammac = gamma(indc);
    if gammac<=-tol
         fprintf('\nsvcm_train error: gamma(indc) = %g <= 0 (Q not positive definite)\n', gammac)
    elseif gammac==Inf
         fprintf('\nsvcm_train error: gamma(indc) = %g (Q rank deficient)\n\n', gammac)
         return
    end
      
    
    % support vector constraints: -C<=a(inds)<=C 
    zlims = (0-z0);                            % by default, immaterial
    if ls>0
         warning off;
         is=find(support==indc);
         warning on;
%is=0;
         if any(is)                              % leave-indc-out, remove from inds
            zlims = 0;                          % clamp z; no change to variables
         else
            betaa = beta(2:ls+1);                % beta terms corresponding to a(inds)  (not b)
            void = (betaa==0);                   % void zero betaa values ...
            if any(any(~void))
                warning off                      % suppress warning div. by 0
                 zlim1 =  C*(betaa*polc>0).*ing;
                 zlim2 = -C*(betaa*polc<0).*ings;
                 zlim1(isnan(zlim1))=0;           % For C Infinite
                 zlim2(isnan(zlim2))=0;
                 zlims = zlim1 + zlim2;
%zlims=0*ones(ls,1);                
                zlims = (zlims - a(support))./betaa;
                warning on              
                zlims(void) = Inf*polc;          % ... which don't enter the constraints
                [zmins, is] = min(zlims*polc,[],1);
                imin = find(zlims*polc==zmins);
                if length(imin)>1
                    [gmax, imax] = max(abs(betaa(imin)),[],1);
                    is = imin(imax);
                end
                zlims = zmins*polc;                   % pick tightest constraintt
            end
        end
    end

    % error vector constraints: g(inde)<=0 and gs(inde)<=0
    zlime = (0-z0);                                 % by default, immaterial
    if le>0
       warning off
       ie = find(inde==indc);
       warning on
%ie=[0];
        if any(ie)                               % leave-indc-out, remove from inde
            zlime = 0;                          % clamp z; no change to variables
        else
           gammae = gamma(inde);
           void = ((gammae*polc<0)|(gammae==0));
                        % void g moving down, or zero gamma...
           if any(any(~void))
                warning off % suppress div. by 0
                zlime = -g(inde)./gammae;
                warning on
                zlime(void) = Inf*polc;    % ... which don't enter the constraints
                [zmine, ie] = min(zlime*polc,[],1);
                imin = find(zlime*polc==zmine);
                if length(imin)>1
                   [gmax, imax] = max(abs(gammae(imin)),[],1);   
                   ie = imin(imax);
                end 
                zlime = zmine*polc;              % pick tightest constraint
            end
        end
     end   
     
     % error vector constraints: g(inde)<=0 and gs(inde)<=0
    zlimes = (0-z0);                               % by default, immaterial
    if les>0
       warning off
       ies = find(indes==indc);
       warning on
%ies=[0];
        if any(ies)                               % leave-indc-out, remove from inde
            zlimes = 0;                          % clamp z; no change to variables
        else
           gammaes = gamma(indes);
           voids = ((gammaes*polc>0)|gammaes==0);
                        % void g moving down, or zero gamma...
           if any(any(~voids))
                warning off % suppress div. by 0
                zlimes = -(2*epsilon-g(indes))./-gammaes;
                warning o
                zlimes(voids) = Inf*polc;  % ... which don't enter the constraints
                [zmines, ies] = min(zlimes*polc,[],1);
                imins = find(abs(zlimes)==zmines);
                if length(imins)>1
                   [gmaxs, imaxs] = max(abs(gammaes(imins)),[],1);
                   ies = imins(imaxs);
                end
                zlimes = zmines*polc;
            end
        end
     end   
          
    % ordinary vector constraints: g(indo)>=0 (only for those that already are)
    zlimo = (0-z0);                            % by default, immaterial
    if lo>0 
        gammao = gamma(indo);
        void=(indo==indc)|(g(indo)<0)|(gammao*polc>0)|(gammao==0)|~processed(indo);
                           % void c, g negative, g moving up, or zero gamma...
                           % ... or, if online, points not seen previously,...
        if any(any(~void))
           warning off % suppress div. by 0
           zlimo = -g(indo)./gammao;
           warning on
           zlimo(void) = Inf*polc;              % ... which don't enter the constraints
           [zmino, io] = min(zlimo*polc,[],1);
           imin = find(zlimo*polc==zmino);
           if length(imin)>1
                 [gmax, imax] = max(abs(gammao(imin)),[],1);
                 io = imin(imax); 
           end
           zlimo = zmino*polc;                  % pick tightest constraint
        end
    end
    
    % ordinary vector constraints:  gs(indo)>=0 (only for those that already are)
    zlimos = (0-z0);                            % by default, immaterial
    if lo>0 
        gammao = gamma(indo);
        void = (indo==indc)|(2*epsilon-g(indo)<0)|(gammao*polc<0)|(gammao==0)|~processed(indo);
                           % void c, g negative, g moving up, or zero gamma...
                           % ... or, if online, points not seen previously,...
        if any(any(~void)) 
            warning off % suppress div. by 0
            zlimos = -(2*epsilon-g(indo))./-gammao;
            warning on
            zlimos(void) = Inf*polc;              % ... which don't enter the constraints
            [zminos, ios] = min(zlimos*polc,[],1);   %segur que zlimos es menor que z0
            imin = find(zlimos==zminos*polc);
            if length(imin)>1
                 [gmax, imax] = max(abs(gammao(imin)),[],1);
                 ios = imin(imax);
            end
            zlimos = zminos*polc;                  % pick tightest constraint
        end
    end
    

    [Nle,Nle2]=size(zlimo);
    if Nle==0
       zlimo=(0-z0);
    end
    [Nle,Nle2]=size(zlimos);
    if Nle==0
       zlimos=(0-z0);
    end

    
    % find constraint-satisfying z
    [z,flag] = min([zlim;zlims;zlime;zlimes;zlimo;zlimos]*polc);
    
    f=find(isnan([zlim;zlims;zlime;zlimes;zlimo;zlimos]));
    if any(f)
       fprintf('NaN')
       return
    end
        
    z=z*polc+z0;
    if (z-z0)*polc<0
        fprintf('\nsvcm_train error: z-z0 of wrong polarity (Q not positive definite)\n\n')
        return
    end
     
%     if (debug==1) & abs(z-z0)<eps & ls>0
%         fprintf('\n%g*', flag)            % procrastinating iteration!  no progress made
%     end
        
    % update a, b, g and W from z-z0
    if ls>0                             % z = a(indc)
        a(indc) = z;
        b = b+(z-z0)*beta(1);
        a(support) = a(support)+(z-z0)*beta(2:ls+1);
        g = g+(z-z0)*gamma;             % update g
     else                                
        b = z;
        g = g+(z-z0)*gamma;
    end
    
    if ((debug==1) & ((sum(a)<-tol)|(sum(a)>tol)))
         fprintf('\nEp!! %g ', sum(a))
    end
    
     
    iter = iter+1;

    % bookkeeping: move elements across indc, inds, inde and indo, and update R and Qs
    converged = (flag<2);               % done with indc; no other changes in inds/inde
    incl_inds = 0;
    if flag==1                          % a(indc) reaches the limits 0 or C, stop moving
        a(indc) = 0*polc;            % should be OK, just to avoid round-of            
         
%     elseif flag==2                      % add indc to support vectors ..
%         incl_inds = 1;
%         indb = indc;                            % ... store in buffer indb for now
%         Qb = Qc;
%            
%     elseif flag==3                      % add indc to support vectors ...
%         incl_inds = 2;
%         indb = indc;                            % ... store in buffer indb for now
%         Qb = Qc;
        
    elseif flag==2                      % one of support vectors becomes error or other vector
        indso = support(is);                    % outgoing inds
%fprintf('Surt %g com support\n', indso);
        free_indc = (indc==indso);              % leave-indc-out: indc is part of inds
        if free_indc
            g(indso) = 0;                       % same
            indo = [indo; indso];
            lo = lo+1;
            ik=find(indso==inds);
            if any(ik) inds=inds([1:ik-1,ik+1:end]); end
            ik=find(indso==indss);
            if any(ik) indss=indss([1:ik-1,ik+1:end]); end  
        elseif beta(is+1)*polc<0                   % a(indso)=0 or indso=indc, move to indo
            if any(indso==inds)
              a(indso) = 0;                   % should be OK, just to avoid round-off
              g(indso) = 0;                       % same
              indo = [indo; indso];
              lo = lo+1;
              ik=find(indso==inds);
              inds=inds([1:ik-1,ik+1:end]);
            else
              a(indso)= -C; 
              g(indso) = 2*epsilon;
              indes=[indes; indso];
              les=les+1;
              ik=find(indso==indss);
              indss=indss([1:ik-1,ik+1:end]);
            end
        else % beta(is+1)*polc>0 & ~free_indc   % a(indso)=C, move to inde
            if any(indso==inds)
              a(indso) = C;                       % should be OK, just to avoid round-off
              g(indso) = 0;                       % same
              inde = [inde; indso];
              le = le+1;
              ik=find(indso==inds);
              inds=inds([1:ik-1,ik+1:end]);
            else
              a(indso) = 0;                   % should be OK, just to avoid round-off
              g(indso) = 2*epsilon;               % same
              indo = [indo; indso];
              lo = lo+1;
              ik=find(indso==indss);
              indss=indss([1:ik-1,ik+1:end]);
            end   
        end
        support = support([1:is-1,is+1:ls]);          % remove from inds
        ings = ings([1:is-1,is+1:ls]);
        ing = ing([1:is-1,is+1:ls]);
        stripped = [1:is,is+2:ls+1];            % also from Qs and R ...
        Qs = Qs(stripped,:);
        ls = ls-1;
        if ls > 0
            if R(is+1,is+1)==0
                fprintf('\nsvcm_train error: divide by zero in R contraction\n')
                R(is+1,is+1)=1e-8;
            end
            R = R(stripped,stripped)-R(stripped,is+1)*R(is+1,stripped)/R(is+1,is+1);
        else % no support vectors left
            R = Inf;
         end
         
      elseif flag==3                       % one of error vectors becomes support/other vector
        indeo = inde(ie);                        % outgoing inde
        if indc==indeo                           % leave-indc-out
            indo = [indo; indeo];                % add inde(ie) to other vectors
            lo = lo+1;
        else
            incl_inds = 1;                       % add inde(ie) to support vectors ...
            indb = indeo;                        % ... store in buffer indb for now
            Qb = kernelsvr(x(indb,:),x);
            Qb(indb) = Qb(indb)+eps2;
            kernelcount = kernelcount+1;
        end
        inde = inde([1:ie-1,ie+1:le]);           % remove from inde
        le = le-1;
        
    elseif flag==4                       % one of error vectors becomes support/other vector
        indeo = indes(ies);                        % outgoing inde
        if indc==indeo                           % leave-indc-out
            indo = [indo; indeo];                % add inde(ie) to other vectors
            lo = lo+1;
        else
            incl_inds = 2;                       % add inde(ie) to support vectors ...
            indb = indeo;                        % ... store in buffer indb for now
            Qb = kernelsvr(x(indb,:),x);
            Qb(indb) = Qb(indb)+eps2;
            kernelcount = kernelcount+1;
        end
        indes = indes([1:ies-1,ies+1:les]);           % remove from inde
        les = les-1;

    elseif flag==5                       % one of other vectors becomes support vector
        indoo = indo(io);                        % outgoing indo
        incl_inds = 1;                           % add indo(io) to support vectors ...
        indb = indoo;                            % ... store in buffer indb for now
        
        Qb = kernelsvr(x(indb,:),x);
        Qb(indb) = Qb(indb)+eps2;
        kernelcount = kernelcount+1;
        
        indo = indo([1:io-1,io+1:lo]);           % remove from indo
        lo = lo-1;
    
    elseif flag==6                       % one of other vectors becomes support vector
        indoo = indo(ios);                        % outgoing indo
        incl_inds = 2;                           % add indo(io) to support vectors ...
        indb = indoo;                            % ... store in buffer indb for now
        
        Qb = kernelsvr(x(indb,:),x);
        Qb(indb) = Qb(indb)+eps2;
        kernelcount = kernelcount+1;
        
        indo = indo([1:ios-1,ios+1:lo]);           % remove from indo
        lo = lo-1;
    end

    if incl_inds ==1|incl_inds==2                % move buffer indb into support vectors inds
%fprintf('Entra %g com support\n', indb);
       if incl_inds==1
           inds = [inds; indb];                     % move to inds ...
           ings=[ings;0];
           ing =[ing;1];
        else
           indss=[indss; indb];  
           ings=[ings;1];
           ing =[ing;0];
        end
        support = [support; indb];
        ls = ls+1;
        Qs = [Qs; Qb];                           % and also Qs and R ...
        if ls==1                                 % compute R directly
            R = [-Qb(indb), 1; 1, 0];
        else                                     % compute R recursively
%             if (flag==2) |(flag==3)                % from indc; use beta and gamma
%                     pivot = gamma(indb);
%             else % flag==4                       % from inde; compute beta and pivot
                beta=-R*Qs(1:ls,indb);
                pivot = [beta',1]*Qs(:,indb);
%           end
            if abs(pivot)<eps2        % should be eps2 when kernel is singular (e.g., linear)
                fprintf('\nsvcm_train error: pivot = %g < %g in R expansion\n\n', pivot, eps2);
                pivot = eps2*polc;
            end
            R = [R,zeros(ls,1);zeros(1,ls+1)]+[beta;1]*[beta',1]/pivot;
            % minor correction in R to avoid numerical instability in recursion when data is near-singular
            if ls>1
      			Qss = [[0;ones(ls,1)],Qs(:,support)];
     				R = R+R'-R*Qss*R';
            end
        end
    end
    
   
    % indc index adjustments (including leave-one-out)
    if (converged) & ls>0             % indc is now part of inde
        i = find(indo==indc);
        indo = indo([1:i-1,i+1:lo]);  % remove indc from indo
        lo = lo-1;
        indnew=[];
    end
    
    % prepare for next iteration, if any
%    free = (g(indo)<-tol|2*epsilon-g(indo)<-tol);    
    
    free = (a(indnew)<-tol|a(indnew)>tol); % free, candidate support or error vector
    left= indnew(free);
%    if ~any(free)
%      free = (g(indo)<-tol|2*epsilon-g(indo)<-tol); % free, candidate support or error vector
%      left = indo(free);                    % candidates left
%    end
    if ~any(free)
        continued = 0;
    end

%if (g(indc)<-tol | 2*epsilon-g(indc)<-tol)
%   fk=indc*ones(1,length(indo))';
%   free = (indc==indo);
%   if ~any(free)
%      free = ~processed(indo)&(g(indo)<-tol|2*epsilon-g(indo)<-tol);
%   end
%else   
%   free = ~processed(indo)&(g(indo)<-tol|2*epsilon-g(indo)<-tol);
%end


%    fprintf('svrm message: indc=%g; ', indc);
%    fprintf('flag=%g; ', flag);
%    fprintf('gs=%g %g; a=%g; ', g(indc), 2*epsilon-g(indc), a(indc));
%    fprintf('lo=%g; ', length(indo));
%    fprintf('ls=%g\n' , length(support));

%if (ls>0)
%      n = size(y,1); m = size(y,1); H = zeros(m,n);  
%   for i=1:m 
%      for j=1:n
%         H(i,j) = svkernel('rbf',y(i,:),y(j,:));
%      end
%   end
%   W = a(support)'*Qs(2:end,support)*a(support) - y'*a + epsilon*sum(abs(a)); 
%   fprintf('Energy1=%g; ', -W)
%   W = a(support)'*H(support,support)*a(support) - y'*a + epsilon*sum(abs(a)); 
%   fprintf('Energy2=%g; ', -W)
%   W = max(a,zeros(length(a),1))'* H *min(a,zeros(length(a),1)) - y'*a + epsilon*sum(abs(a)); 
%   fprintf('Energy3=%g; Sum alpha=%g; \n\n', -W, sum(a))
%end

if verbose==1
    disp("VERBOSEEEEEEE")
    disp(verbose)
    f = find(isnan(a));
    if any(f)
       fprintf('svcm_train error: a(%g) = %g\n', [f, a(f)]')
    end
    f = find(isnan(g));
    if any(f)
       fprintf('svcm_train error: g(%g) = %g\n', [f, g(f)]')
    end
    f = find(a>C|a<-C);
    if any(f)
       fprintf('svcm_train error: a(%g) = %g > C o <-C\n', [f, a(f)]')
%          a(f)=0;
    end
    f = inde(find(a(indes)>-C));
    if any(f)
            fprintf('svcm_train error: a(%g) = %g ~= C\n', [f, a(f)]')
    end
    f = inde(find(a(inde)<C));
    if any(f)
            fprintf('svcm_train error: a(%g) = %g ~= C\n', [f, a(f)]')
    end   
%        if abs(y'*a)>tol
%                fprintf('svcm_train error: y''*a = %g ~= 0 (tol=%g)\n', y'*a, tol);
%        end
    f = indo(find((abs(a(indo))<tol)&(a(indo)~=0)));
    if any(f)
        fprintf('svcm_train error: a(%g) = %g ~= 0 (tol=%g)\n', [f, a(f), f*0+tol]');
%            a(f)=0;
    end
%        f = inds(find(abs(g(inds))>tol));
%        if any(f)
%            fprintf('svcm_train error: g(%g) = %g ~= 0 (tol=%g)\n', [f, g(f), f*0+tol]');
%            g(f)=0;
%        end
%        f = indss(find(abs(2*epsilon-g(indss)))>tol);
%        if any(f)
%           fprintf('svcm_train error: g(%g) = %g ~= %g (tol=%g)\n', [f, g(f), 2*epsilon*ones(length(f)), f*0+tol]');
%           g(f)=2*epsilon;
%        end
    f = inde(find(g(inde)>tol));
    if any(f)
        fprintf('svcm_train error: g(%g) = %g > 0 (tol=%g)\n', [f, g(f), f*0+tol]');
    end
%        inda = sort([indo;inds;inde;indss;indes]);
%        if any(inda~=(1:L)')
%            fprintf('svcm_train error: union [indo;inds;inde] does not equate entire set\n');
%        end
    if ls~=length(inds)+length(indss)
        fprintf('svcm_train error: miscount in number of support vectors\n');
    end
    if le~=length(inde)
        fprintf('svcm_train error: miscount in number of error vectors\n');
    end
    if lo~=length(indo)
        fprintf('svcm_train error: miscount in number of other vectors\n');
    end
 end
                                           % candidate support/error vectors in indoc

                                           
svmState.a = a;
svmState.b = b;
svmState.g = g;
svmState.inds = inds; 
svmState.indss = indss;
svmState.inde = inde;
svmState.indes = indes;
svmState.support = support;
svmState.ing = ing;
svmState.ings = ings;
svmState.indo = indo;
svmState.R = R;
svmState.Qs = Qs;
svmState.Qc = Qc;
svmState.processed = processed;
svmState.eps = eps;
svmState.eps2 = eps2;
svmState.tol = tol;
end

