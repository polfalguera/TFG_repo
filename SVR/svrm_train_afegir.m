function [svmState, continued] = svrm_train_afegir(svmState, xold, yold, x, y, C, epsilon);  
% Add epsilon - MM 14/11/01
% function [svmState.a, svmState.b, svmState.g, svmState.inds, svmState.inde, indw] = svcm_train(x, y, C);
%        svmState.support vector classification machine
%        incremental learning, and leave-one-out cross-validation
%        soft margin
%        uses "kernel.m"
%
%        x: independent variable, (L,N) with L: number of points; N: dimension
%        y: dependent variable, (L,1) containing class labels (-1 or +1)
%        C: soft-margin regularization constant
%
%        svmState.a: alpha coefficients 
%        svmState.b: offset coefficient  
%        svmState.g: derivatives (adding one yields margins for each point)
%        svmState.inds: indices of svmState.support vectors in S
%        svmState.indss: indices of svmState.support vectors in S*
%        svmState.inde: indices of error vectors in E
%        svmState.indes: indices of error star vectors 
%%%%%%%%%% version 0.9; last revised 02/12/2001; send comments to gert@jhu.edu %%%%%%%%%%

global verbose
global debug
%global svmState.a svmState.b svmState.g svmState.inds svmState.indss svmState.inde svmState.indes svmState.support svmState.ing svmState.ings svmState.indo svmState.R svmState.Qs svmState.Qc svmState.processed svmState.eps svmState.eps2 svmState.tol

[L,N] = size(x);
[Lold,Nold] = size(xold);
svmState.indo = [(L+Lold:-1:Lold+1)'; svmState.indo];
svmState.a = [svmState.a ; zeros(L,1)];   % coefficients, sparse

for i=1:Lold,
   Qsn(i,:) = svmState.a(i).*kernelsvr(xold(i,:),x);
end 
gnew = -y+(epsilon+svmState.eps+svmState.b)*ones(L,1)+ sum(Qsn)';       % derivative of energy function
svmState.g = [svmState.g; gnew];

svmState.processed = [ones(Lold,1); zeros(L,1)];
clear Qsn
Qsn(1,:) = ones(L,1)';
for i=1:length(svmState.support),
 Qsn(i+1,:) = kernelsvr(xold(svmState.support(i),:),x);
end 
svmState.Qs = [svmState.Qs Qsn];
x= [xold ; x];
y= [yold; y];

%fprintf('Support vector soft-margin regression with incremental learning\n')
%fprintf('  %svmState.g training points\n', L)
%fprintf('  %svmState.g dimensions\n\n', N)
[L,N] = size(x);

ls = length(svmState.inds)+length(svmState.indss);  % number of svmState.support vectors;
le = length(svmState.inde);                % number of error vectors;
les = length(svmState.indes);              % number of error vectors; - MM 14/11/01
la = ls+le+les;                   % both                     - MM 14/11/01
lo = length(svmState.indo);                % number of other vectors;

kernelcount = 0;
iter = 0;                         % iteration count

indc = 0;                             % candidate vector
indso = 0;                            % svmState.a recycled svmState.support vector; used as buffer
free = svmState.g(svmState.indo)<0|2*epsilon-svmState.g(svmState.indo)<0; % free, candidate svmState.support or error vector
left = svmState.indo(free);                    % candidates left
continued = any(left);

while continued                   % check for remaining free points or leave-one-outs to process
    
    % select candidate indc
    indc_prev = indc;
    
    indc = left(length(left));        % take top of the stack, "last-in, first-out
    
    if indc_prev>0
%        svmState.processed(1:indc_prev) = 1;   % record last and all preceding point
         svmState.processed(1:indc-1)=1;
    end       
       
    
    if verbose==1
       if indc~=indc_prev
            fprintf('\n%g',indc)
       else
            fprintf('o')
       end
    end   
    
%    if indc==5
%       fprintf('')
%    end


    % get svmState.Qc, row of hessian corresponding to indc (needed for gamma)
    xc = x(indc,:);
    yc = y(indc);
    svmState.Qc = kernelsvr(xc,x);            %  En classificació  svmState.Qc = (yc*y').*kernel(xc,x);
    svmState.Qc(indc) = svmState.Qc(indc)+svmState.eps2;
    
    % prepare to increment/decrement z = svmState.a(indc)' or y(indc)*svmState.b, subject to constraints.
    % move z up when adding indc ((re-)training), down when removing indc (leave-one-out or svmState.g>0)
    upc = (svmState.g(indc)<0) & ~(2*epsilon-svmState.g(indc)<0);
    polc = 2*upc-1;                % polarity of increment in z
    beta = -svmState.R*svmState.Qs(:,indc);          % change in [svmState.b;svmState.a(svmState.inds)] per change in svmState.a(indc)
    if ls>0
        % move z = svmState.a(indc)'
        gamma = svmState.Qc'+svmState.Qs'*beta;      % change in svmState.g(:) per change in z = svmState.a(indc)'
        z0 = svmState.a(indc);              % initial z value
        zlim = C*polc;             % constraint on svmState.a(indc)
    else % ls==0
        % move z = svmState.b and keep svmState.a(indc) constant; there is no svmState.a(:) free to move in svmState.inds!
        gamma = svmState.Qs';               % change in svmState.g(:) per change in z = svmState.b
        z0 = svmState.b;                    % initial z value
        zlim = Inf*polc;           % no constraint on svmState.b
    end
    
    gammac = gamma(indc);
    if gammac<=-svmState.tol
         fprintf('\nsvcm_train error: gamma(indc) = %g <= 0 (Q not positive definite)\n', gammac)
    elseif gammac==Inf
         fprintf('\nsvcm_train error: gamma(indc) = %g (Q rank deficient)\n\n', gammac)
         return
    end
      
    % intrinsic limit: svmState.g(indc) = 0, where indc becomes svmState.support vector
    if (polc==1)
          zlimc = z0-svmState.g(indc)'./gammac;
    else
          zlimc = Inf*polc;
    end
     
    % intrinsic limit: gs(indc) = 0, where indc becomes svmState.support vector
    if (polc==-1)
          zlimcs = z0-(2*epsilon-svmState.g(indc))'./-gammac;
    else
          zlimcs = Inf*polc;
    end

    % svmState.support vector constraints: -C<=svmState.a(svmState.inds)<=C 
    zlims = Inf*polc;                            % by default, immaterial
    if ls>0
        warning off;
        is = any(svmState.inds==indc)|any(svmState.indss==indc);
        warning on;
        if is                                    % leave-indc-out, remove from svmState.inds
            zlims = z0;                          % clamp z; no change to variables
         else
            betaa = beta(2:ls+1);                % beta terms corresponding to svmState.a(svmState.inds)  (not svmState.b)
            void = (betaa==0);                   % void zero betaa values ...
            if any(any(~void))
                warning off                      % suppress warning div. by 0
                zlim1 =  C*(betaa*polc>0).*svmState.ing;
                zlim2 = -C*(betaa*polc<0).*svmState.ings;
                zlim1(isnan(zlim1))=0;
                zlim2(isnan(zlim2))=0;
                zlims = zlim1 + zlim2;
                zlims = z0+(zlims - svmState.a(svmState.support))./betaa;
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

    % error vector constraints: svmState.g(svmState.inde)<=0 and gs(svmState.inde)<=0
    zlime = Inf*polc;                                 % by default, immaterial
    if le>0
       warning off
       ie = any(svmState.inde==indc);
       warning on
        if any(ie)                               % leave-indc-out, remove from svmState.inde
            zlime = z0;                          % clamp z; no change to variables
        else
           gammae = gamma(svmState.inde);
           void = ((gammae*polc<0)|(gammae==0));
                        % void svmState.g moving down, or zero gamma...
           if any(any(~void))
                warning off % suppress div. by 0
                zlime = z0-svmState.g(svmState.inde)./gammae;
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
     
     % error vector constraints: svmState.g(svmState.inde)<=0 and gs(svmState.inde)<=0
    zlimes = Inf*polc;                               % by default, immaterial
    if les>0
       warning off
       ies = any(svmState.indes==indc);
       warning on
        if any(ies)                               % leave-indc-out, remove from svmState.inde
            zlimes = z0;                          % clamp z; no change to variables
        else
           gammaes = gamma(svmState.indes);
           voids = ((gammaes*polc>0)|gammaes==0);
                        % void svmState.g moving down, or zero gamma...
           if any(any(~voids))
                warning off % suppress div. by 0
                zlimes = z0-(2*epsilon-svmState.g(svmState.indes))./-gammaes;
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
          
    % ordinary vector constraints: svmState.g(svmState.indo)>=0 (only for those that already are)
    zlimo = Inf*polc;                            % by default, immaterial
    if lo>0 
        gammao = gamma(svmState.indo);
        void=(svmState.indo==indc)|(svmState.g(svmState.indo)<0)|(gammao*polc>0)|(gammao==0)|~svmState.processed(svmState.indo);
                           % void c, svmState.g negative, svmState.g moving up, or zero gamma...
                           % ... or, if online, points not seen previously,...
        if any(any(~void))
           warning off % suppress div. by 0
           zlimo = z0-svmState.g(svmState.indo)./gammao;
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
    
    % ordinary vector constraints:  gs(svmState.indo)>=0 (only for those that already are)
    zlimos = Inf*polc;                            % by default, immaterial
    if lo>0 
        gammao = gamma(svmState.indo);
        void = (svmState.indo==indc)|(2*epsilon-svmState.g(svmState.indo)<0)|(gammao*polc<0)|(gammao==0)|~svmState.processed(svmState.indo);
                           % void c, svmState.g negative, svmState.g moving up, or zero gamma...
                           % ... or, if online, points not seen previously,...
        if any(any(~void)) 
            warning off % suppress div. by 0
            zlimos = z0-(2*epsilon-svmState.g(svmState.indo))./-gammao;
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
       zlimo=Inf*polc;
    end
    [Nle,Nle2]=size(zlimos);
    if Nle==0
       zlimos=Inf*polc;
    end

    
    % find constraint-satisfying z
    [z,flag] = min([zlim;zlimc;zlimcs;zlims;zlime;zlimes;zlimo;zlimos]*polc);
    
    f=find(isnan([zlim;zlimc;zlimcs;zlims;zlime;zlimes;zlimo;zlimos]));
    if any(f)
       fprintf('NaN')
       return
    end
        
    z=z*polc;
    if (z-z0)*polc<0
        fprintf('\nsvcm_train error: z-z0 of wrong polarity (Q not positive definite)\n\n')
        return
    end
     
%     if (debug==1) & abs(z-z0)<svmState.eps & ls>0
%         fprintf('\n%g*', flag)            % procrastinating iteration!  no progress made
%     end
        
    % update svmState.a, svmState.b, svmState.g and W from z-z0
    if ls>0                             % z = svmState.a(indc)
        svmState.a(indc) = z;
        svmState.b = svmState.b+(z-z0)*beta(1);
        svmState.a(svmState.support) = svmState.a(svmState.support)+(z-z0)*beta(2:ls+1);
        svmState.g = svmState.g+(z-z0)*gamma;             % update svmState.g
     else                                
        svmState.b = z;
        svmState.g = svmState.g+(z-z0)*gamma;
    end
    
    if ((debug==1) & ((sum(svmState.a)<-svmState.tol)|(sum(svmState.a)>svmState.tol)))
         fprintf('\nEp!! %g ', sum(svmState.a))
    end
    
     
    iter = iter+1;

    % bookkeeping: move elements across indc, svmState.inds, svmState.inde and svmState.indo, and update svmState.R and svmState.Qs
    converged = (flag<4);               % done with indc; no other changes in svmState.inds/svmState.inde
    incl_inds = 0;
    if flag==1                          % svmState.a(indc) reaches the limits 0 or C, stop moving
        if upc                                  % svmState.a(indc)=C, add to svmState.inde
            svmState.inde = [svmState.inde; indc];
            le = le+1;
            svmState.a(indc) = C*polc;            % should be OK, just to avoid round-of        
        else
            svmState.indes = [svmState.indes; indc];
            les = les+1;
            svmState.a(indc) = C*polc;            % should be OK, just to avoid round-of        
         end    
         
    elseif flag==2                      % add indc to svmState.support vectors ..
        incl_inds = 1;
        indb = indc;                            % ... store in buffer indb for now
        Qb = svmState.Qc;
           
    elseif flag==3                      % add indc to svmState.support vectors ...
        incl_inds = 2;
        indb = indc;                            % ... store in buffer indb for now
        Qb = svmState.Qc;
        
    elseif flag==4                      % one of svmState.support vectors becomes error or other vector
       indso = svmState.support(is);                    % outgoing svmState.inds
%fprintf('Surt %g com svmState.support\n', indso);
       free_indc = (indc==indso);              % leave-indc-out: indc is part of svmState.inds
       if beta(is+1)*polc<0 | free_indc        % svmState.a(indso)=0 or indso=indc, move to svmState.indo
         if any(indso==svmState.inds)  
            if ~free_indc
%                if svmState.a(indso)>C/2
%                    fprintf('svcm_train error: svmState.a(indso)=%g; 0 anticipated\n', svmState.a(indso));
%                end
                svmState.a(indso) = 0;                   % should be OK, just to avoid round-off
            end
            svmState.g(indso) = 0;                       % same
            svmState.indo = [svmState.indo; indso];
            lo = lo+1;
            ik=find(indso==svmState.inds);
            svmState.inds=svmState.inds([1:ik-1,ik+1:end]);
         else
            svmState.a(indso)= -C; 
            svmState.g(indso) = 2*epsilon;
            svmState.indes=[svmState.indes; indso];
            les=les+1;
            ik=find(indso==svmState.indss);
            svmState.indss=svmState.indss([1:ik-1,ik+1:end]);
         end
       else % beta(is+1)*polc>0 & ~free_indc   % svmState.a(indso)=C, move to svmState.inde
         if any(indso==svmState.inds)  
%            if svmState.a(indso)<C/2
%                fprintf('svcm_train error: svmState.a(indso)=%g; C anticipated\n', svmState.a(indso));
%            end
            svmState.a(indso) = C;                       % should be OK, just to avoid round-off
            svmState.g(indso) = 0;                       % same
            svmState.inde = [svmState.inde; indso];
            le = le+1;
            ik=find(indso==svmState.inds);
            svmState.inds=svmState.inds([1:ik-1,ik+1:end]);
         else
            if ~free_indc
%                if svmState.a(indso)>C/2
%                    fprintf('svcm_train error: svmState.a(indso)=%g; 0 anticipated\n', svmState.a(indso));
%                end
                svmState.a(indso) = 0;                   % should be OK, just to avoid round-off
            end
            svmState.g(indso) = 2*epsilon;               % same
            svmState.indo = [svmState.indo; indso];
            lo = lo+1;
            ik=find(indso==svmState.indss);
            svmState.indss=svmState.indss([1:ik-1,ik+1:end]);
         end   
        end
        svmState.support = svmState.support([1:is-1,is+1:ls]);          % remove from svmState.inds
        svmState.ings = svmState.ings([1:is-1,is+1:ls]);
        svmState.ing = svmState.ing([1:is-1,is+1:ls]);
        stripped = [1:is,is+2:ls+1];            % also from svmState.Qs and svmState.R ...
        svmState.Qs = svmState.Qs(stripped,:);
        ls = ls-1;
        if ls > 0
            if svmState.R(is+1,is+1)==0
                fprintf('\nsvcm_train error: divide by zero in svmState.R contraction\n')
                svmState.R(is+1,is+1)=1e-8;
            end
            svmState.R = svmState.R(stripped,stripped)-svmState.R(stripped,is+1)*svmState.R(is+1,stripped)/svmState.R(is+1,is+1);
        else % no svmState.support vectors left
            svmState.R = Inf;
         end
         
      elseif flag==5                       % one of error vectors becomes svmState.support/other vector
        indeo = svmState.inde(ie);                        % outgoing svmState.inde
        if indc==indeo                           % leave-indc-out
            svmState.indo = [svmState.indo; indeo];                % add svmState.inde(ie) to other vectors
            lo = lo+1;
        else
            incl_inds = 1;                       % add svmState.inde(ie) to svmState.support vectors ...
            indb = indeo;                        % ... store in buffer indb for now
            Qb = kernelsvr(x(indb,:),x);
            Qb(indb) = Qb(indb)+svmState.eps2;
            kernelcount = kernelcount+1;
        end
        svmState.inde = svmState.inde([1:ie-1,ie+1:le]);           % remove from svmState.inde
        le = le-1;
        
    elseif flag==6                       % one of error vectors becomes svmState.support/other vector
        indeo = svmState.indes(ies);                        % outgoing svmState.inde
        if indc==indeo                           % leave-indc-out
            svmState.indo = [svmState.indo; indeo];                % add svmState.inde(ie) to other vectors
            lo = lo+1;
        else
            incl_inds = 2;                       % add svmState.inde(ie) to svmState.support vectors ...
            indb = indeo;                        % ... store in buffer indb for now
            Qb = kernelsvr(x(indb,:),x);
            Qb(indb) = Qb(indb)+svmState.eps2;
            kernelcount = kernelcount+1;
        end
        svmState.indes = svmState.indes([1:ies-1,ies+1:les]);           % remove from svmState.inde
        les = les-1;

    elseif flag==7                       % one of other vectors becomes svmState.support vector
        indoo = svmState.indo(io);                        % outgoing svmState.indo
        incl_inds = 1;                           % add svmState.indo(io) to svmState.support vectors ...
        indb = indoo;                            % ... store in buffer indb for now
        
        Qb = kernelsvr(x(indb,:),x);
        Qb(indb) = Qb(indb)+svmState.eps2;
        kernelcount = kernelcount+1;
        
        svmState.indo = svmState.indo([1:io-1,io+1:lo]);           % remove from svmState.indo
        lo = lo-1;
    
    elseif flag==8                       % one of other vectors becomes svmState.support vector
        indoo = svmState.indo(ios);                        % outgoing svmState.indo
        incl_inds = 2;                           % add svmState.indo(io) to svmState.support vectors ...
        indb = indoo;                            % ... store in buffer indb for now
        
        Qb = kernelsvr(x(indb,:),x);
        Qb(indb) = Qb(indb)+svmState.eps2;
        kernelcount = kernelcount+1;
        
        svmState.indo = svmState.indo([1:ios-1,ios+1:lo]);           % remove from svmState.indo
        lo = lo-1;
    end

    if incl_inds ==1|incl_inds==2                % move buffer indb into svmState.support vectors svmState.inds
%fprintf('Entra %g com svmState.support\n', indb);
       if incl_inds==1
           svmState.inds = [svmState.inds; indb];                     % move to svmState.inds ...
           svmState.ings=[svmState.ings;0];
           svmState.ing =[svmState.ing;1];
        else
           svmState.indss=[svmState.indss; indb];  
           svmState.ings=[svmState.ings;1];
           svmState.ing =[svmState.ing;0];
        end
        svmState.support = [svmState.support; indb];
        ls = ls+1;
        svmState.Qs = [svmState.Qs; Qb];                           % and also svmState.Qs and svmState.R ...
        if ls==1                                 % compute svmState.R directly
            svmState.R = [-Qb(indb), 1; 1, 0];
        else                                     % compute svmState.R recursively
            if (flag==2) |(flag==3)                % from indc; use beta and gamma
                    pivot = gamma(indb);
            else % flag==4                       % from svmState.inde; compute beta and pivot
                beta=-svmState.R*svmState.Qs(1:ls,indb);
                pivot = [beta',1]*svmState.Qs(:,indb);
            end
            if abs(pivot)<svmState.eps2        % should be svmState.eps2 when kernel is singular (e.g., linear)
                fprintf('\nsvcm_train error: pivot = %g < %g in svmState.R expansion\n\n', pivot, svmState.eps2);
                pivot = svmState.eps2*polc;
            end
            svmState.R = [svmState.R,zeros(ls,1);zeros(1,ls+1)]+[beta;1]*[beta',1]/pivot;
            % minor correction in svmState.R to avoid numerical instability in recursion when data is near-singular
            if ls>1
      			Qss = [[0;ones(ls,1)],svmState.Qs(:,svmState.support)];
     				svmState.R = svmState.R+svmState.R'-svmState.R*Qss*svmState.R';
            end
        end
    end
    
   
    % indc index adjustments (including leave-one-out)
    if (converged) & ls>0             % indc is now part of svmState.inde
        i = find(svmState.indo==indc);
        svmState.indo = svmState.indo([1:i-1,i+1:lo]);  % remove indc from svmState.indo
        lo = lo-1;
    end
    
    % prepare for next iteration, if any
    free = (svmState.g(svmState.indo)<-svmState.tol|2*epsilon-svmState.g(svmState.indo)<-svmState.tol);    
    
%if (svmState.g(indc)<-svmState.tol | 2*epsilon-g(indc)<-svmState.tol)
%   fk=indc*ones(1,length(svmState.indo))';
%   free = (indc==svmState.indo);
%   if ~any(free)
%      free = ~svmState.processed(svmState.indo)&(svmState.g(svmState.indo)<-svmState.tol|2*epsilon-svmState.g(svmState.indo)<-svmState.tol);
%   end
%else   
%   free = ~svmState.processed(svmState.indo)&(svmState.g(svmState.indo)<-svmState.tol|2*epsilon-svmState.g(svmState.indo)<-svmState.tol);
%end


%    fprintf('svrm message: indc=%g; ', indc);
%    fprintf('flag=%g; ', flag);
%    fprintf('gs=%g %g; svmState.a=%g; ', g(indc), 2*epsilon-g(indc), svmState.a(indc));
%    fprintf('lo=%g; ', length(svmState.indo));
%    fprintf('ls=%g\n' , length(svmState.support));

%if (ls>0)
%      n = size(y,1); m = size(y,1); H = zeros(m,n);  
%   for i=1:m 
%      for j=1:n
%         H(i,j) = svkernel('rbf',y(i,:),y(j,:));
%      end
%   end
%   W = svmState.a(svmState.support)'*svmState.Qs(2:end,svmState.support)*svmState.a(svmState.support) - y'*svmState.a + epsilon*sum(abs(svmState.a)); 
%   fprintf('Energy1=%g; ', -W)
%   W = svmState.a(svmState.support)'*H(svmState.support,svmState.support)*svmState.a(svmState.support) - y'*svmState.a + epsilon*sum(abs(svmState.a)); 
%   fprintf('Energy2=%g; ', -W)
%   W = max(svmState.a,zeros(length(svmState.a),1))'* H *min(svmState.a,zeros(length(svmState.a),1)) - y'*svmState.a + epsilon*sum(abs(svmState.a)); 
%   fprintf('Energy3=%g; Sum alpha=%g; \n\n', -W, sum(svmState.a))
%end

if verbose==1
        f = find(isnan(svmState.a));
        if any(f)
           fprintf('svcm_train error: svmState.a(%g) = %g\n', [f, svmState.a(f)]')
        end
        f = find(isnan(svmState.g));
        if any(f)
           fprintf('svcm_train error: g(%g) = %g\n', [f, svmState.g(f)]')
        end
        f = find(svmState.a>C|svmState.a<-C);
        if any(f)
           fprintf('svcm_train error: svmState.a(%g) = %g > C o <-C\n', [f, svmState.a(f)]')
 %          svmState.a(f)=0;
        end
        f = svmState.inde(find(svmState.a(svmState.indes)>-C));
        if any(f)
                fprintf('svcm_train error: svmState.a(%g) = %g ~= C\n', [f, svmState.a(f)]')
        end
        f = svmState.inde(find(svmState.a(svmState.inde)<C));
        if any(f)
                fprintf('svcm_train error: svmState.a(%g) = %g ~= C\n', [f, svmState.a(f)]')
        end   
%        if abs(y'*svmState.a)>svmState.tol
%                fprintf('svcm_train error: y''*svmState.a = %g ~= 0 (svmState.tol=%g)\n', y'*svmState.a, svmState.tol);
%        end
        f = svmState.indo(find((abs(svmState.a(svmState.indo))<svmState.tol)&(svmState.a(svmState.indo)~=0)));
        if any(f)
            fprintf('svcm_train error: svmState.a(%g) = %g ~= 0 (svmState.tol=%g)\n', [f, svmState.a(f), f*0+svmState.tol]');
%            svmState.a(f)=0;
        end
%        f = svmState.inds(find(abs(svmState.g(svmState.inds))>svmState.tol));
%        if any(f)
%            fprintf('svcm_train error: svmState.g(%g) = %g ~= 0 (svmState.tol=%g)\n', [f, svmState.g(f), f*0+svmState.tol]');
%            svmState.g(f)=0;
%        end
%        f = svmState.indss(find(abs(2*epsilon-svmState.g(svmState.indss)))>svmState.tol);
%        if any(f)
%           fprintf('svcm_train error: svmState.g(%g) = %g ~= %g (svmState.tol=%g)\n', [f, svmState.g(f), 2*epsilon*ones(length(f)), f*0+svmState.tol]');
%           svmState.g(f)=2*epsilon;
%        end
        f = svmState.inde(find(svmState.g(svmState.inde)>svmState.tol));
        if any(f)
            fprintf('svcm_train error: svmState.g(%g) = %g > 0 (svmState.tol=%g)\n', [f, svmState.g(f), f*0+svmState.tol]');
        end
        inda = sort([svmState.indo;svmState.inds;svmState.inde;svmState.indss;svmState.indes]);
        if any(inda~=(1:L)')
            fprintf('svcm_train error: union [svmState.indo;svmState.inds;svmState.inde] does not equate entire set\n');
        end
        if ls~=length(svmState.inds)+length(svmState.indss)
            fprintf('svcm_train error: miscount in number of svmState.support vectors\n');
        end
        if le~=length(svmState.inde)
            fprintf('svcm_train error: miscount in number of error vectors\n');
        end
        if lo~=length(svmState.indo)
            fprintf('svcm_train error: miscount in number of other vectors\n');
        end
end
                                           % candidate svmState.support/error vectors in indoc
    if any(free)
        left = svmState.indo(free);                % candidates left, keep (re-)training
    else % ~any(free)                      % done; finish up and (re-)initiate leave-one-out
        continued = 0;
     end
end

