%% Define State Matrices
clear
clc

A1=[1 0 2 0; 0 1 0 2; 0 0 3 0; 0 0 0 3];
B1=[2 0;0 2;3 0;0 3];
x01=[-10;10;-1;1];

A2=[1 0 3 0; 0 1 0 3; 0 0 7 0; 0 0 0 7];
B2=[3 0; 0 3; 7 0; 0 7];
x02=[10;10;1;1];

A3=[1 0 1 0; 0 1 0 1; 0 0 1.1 0; 0 0 0 1.1];
B3=[1 0; 0 1; 1.1 0; 0 1.1];
x03=[10;-10;1;-1];

A4=[1 0 6 0; 0 1 0 6; 0 0 20 0; 0 0 0 20];
B4=[6 0;0 6;20 0; 0 20];
x04=[-10;-10;-1;-1];

%Define global variables
Tf=5;
umax=32;
nx=length(A4);nu=width(B4);
na=4;%number of agents

%% Obtain Prediction Model Matrices
%Phi Matrix of each agent
Phi.a1=computePhi(A1,Tf);Phi.a2=computePhi(A2,Tf);
Phi.a3=computePhi(A3,Tf);Phi.a4=computePhi(A4,Tf);
%Gamma Matrix of each agent
Gam.a1=computeGamma(A1,B1,Tf);Gam.a2=computeGamma(A2,B2,Tf);
Gam.a3=computeGamma(A3,B3,Tf);Gam.a4=computeGamma(A4,B4,Tf);
%% Centralized Optimization
%Generate local cost functions
[H1,c1]=genCost(Phi.a1,Gam.a1,x01,nx,nu,Tf);
[H2,c2]=genCost(Phi.a2,Gam.a2,x02,nx,nu,Tf);
[H3,c3]=genCost(Phi.a3,Gam.a3,x03,nx,nu,Tf);
[H4,c4]=genCost(Phi.a4,Gam.a4,x04,nx,nu,Tf);
%Generate ubound constraints
[LHS,RHS] = ubound(nu,umax,Tf);
%Generate final state matrices
[Af1, bf1] = xfMat(A1,B1,Tf);
[Af2, bf2] = xfMat(A2,B2,Tf);
[Af3, bf3] = xfMat(A3,B3,Tf);
[Af4, bf4] = xfMat(A4,B4,Tf);
%Concatenate H matrices
cent.H=blkdiag(H1,H2,H3,H4);
%Concatenate linear terms
cent.c=[c1,c2,c3,c4];
%Concatenate ubound constraints
cent.LHS=blkdiag(LHS,LHS,LHS,LHS);
cent.RHS=[RHS;RHS;RHS;RHS];
%Centralized equality constraints
cent.Aeq=[bf1,-bf2,zeros(nx,nu*Tf),zeros(nx,nu*Tf);...
          bf1,zeros(nx,nu*Tf),-bf3,zeros(nx,nu*Tf);
          bf1,zeros(nx,nu*Tf),zeros(nx,nu*Tf),-bf4];
cent.beq=[-Af1*x01+Af2*x02;...
          -Af1*x01+Af3*x03;...
          -Af1*x01+Af4*x04];

%SOLVE CENTRALIZED PROBLEM
options = optimoptions('quadprog','Display','off');
cent.us = quadprog(2*cent.H, cent.c,...
                    cent.LHS, cent.RHS,...
                    cent.Aeq, cent.beq,...
                    [],[],[],options);
%% Simulate Trajectories (centralized)
cent.us1=cent.us(1:10,:);cent.us2=cent.us(11:20,:);
cent.us3=cent.us(21:30,:);cent.us4=cent.us(31:40,:);
usf=[cent.us1,cent.us2,cent.us3,cent.us4];
x0f=[x01,x02,x03,x04];
plot_agent_trajectories(Phi, Gam, usf, x0f, Tf)
%% Construct Affine Hull
% Lower and upper bound on input trajectories
utrajmax = (-umax/Tf*ones(1,Tf*2))';
utrajmin= (umax/Tf*ones(1,Tf*2))';
utrajmin1max2=diag(repmat([1; -1], ceil(Tf*2/2), 1))*utrajmax;
utrajmax1min2=diag(repmat([-1; 1], ceil(Tf*2/2), 1))*utrajmax;
%Convex Hull Vertices for Node 1
xflblb_1 = Af1*x01+bf1*utrajmin;
xfubub_1 = Af1*x01+bf1*utrajmax;
xflbub_1 = Af1*x01+bf1*utrajmin1max2;
xfublb_1 = Af1*x01+bf1*utrajmax1min2;

% Convex Hull Vertices for Node 2
xflblb_2 = Af2 * x02 + bf2 * utrajmin;
xfubub_2 = Af2 * x02 + bf2 * utrajmax;
xflbub_2 = Af2 * x02 + bf2 * utrajmin1max2;
xfublb_2 = Af2 * x02 + bf2 * utrajmax1min2;

% Convex Hull Vertices for Node 3
xflblb_3 = Af3 * x03 + bf3 * utrajmin;
xfubub_3 = Af3 * x03 + bf3 * utrajmax;
xflbub_3 = Af3 * x03 + bf3 * utrajmin1max2;
xfublb_3 = Af3 * x03 + bf3 * utrajmax1min2;

% Convex Hull Vertices for Node 4
xflblb_4 = Af4 * x04 + bf4 * utrajmin;
xfubub_4 = Af4 * x04 + bf4 * utrajmax;
xflbub_4 = Af4 * x04 + bf4 * utrajmin1max2;
xfublb_4 = Af4 * x04 + bf4 * utrajmax1min2;

%Vertices
vert1=[xflblb_1, xfubub_1, xflbub_1, xfublb_1];
vert2=[xflblb_2, xfubub_2, xflbub_2, xfublb_2];
vert3=[xflblb_3, xfubub_3, xflbub_3, xfublb_3];
vert4=[xflblb_4, xfubub_4, xflbub_4, xfublb_4];

%% Consensus Algorithm
Hvals={H1,H2,H3,H4};cvals={c1,c2,c3,c4};
Aeqvals={bf1,bf2,bf3,bf4};beqvals={Af1*x01,Af2*x02,Af3*x03,Af4*x04};
error_sequences={};
%Consensus parameters
W=[0.75,0.25,0,0;
   0.25,0.5,0.25,0;
   0,0.25,0.5,0.25;
   0,0,0.25,0.75];
error_sequences = {};
dxf_seq={};
phi_vals=[1,5,10,20];
%Iterations Parameters
N=50;
options = optimoptions('quadprog','Display','off');
for phi=phi_vals
    Wpow=W^phi;
    thmat1=zeros(nx,N);
    thmat2=zeros(nx,N);
    thmat3=zeros(nx,N);
    thmat4=zeros(nx,N);
    err = inf;
    err_ev = []; % Error and error evolution
    dxf=inf;
    dxf_ev=[];
    k=2;
    alpha=0.0001;
    beta=0;%0.02;
    gamma=0.9;
    while k<N
        %ah=(800*alpha)/sqrt(k+800);%Variable step size
        ah=0.05/sqrt(1000*k);
        
        % SOLUTION FOR NODE 1 Given Theta
        th1 = thmat1(:,k); % Theta
        [q1,~,~,~,lm1] = quadprog(2*Hvals{1}, cvals{1}, ...
                                  LHS, RHS, Aeqvals{1}, th1-beqvals{1}, ...
                                  [], [], [], options);
        noise1 = beta * randn(nx, 1);
        subg1 = th1 + ah * lm1.eqlin + noise1 ...
                + gamma * (thmat1(:,k) - thmat1(:,k-1)); % Subgradient component 1
        subg1 = subg1 / norm(subg1);
        xf1 = Af1 * x01 + bf1 * q1;
        
        % SOLUTION FOR NODE 2 Given Theta
        th2 = thmat2(:,k); % Theta
        [q2,~,~,~,lm2] = quadprog(2*Hvals{2}, cvals{2}, ...
                                  LHS, RHS, Aeqvals{2}, th2-beqvals{2}, ...
                                  [], [], [], options);
        noise2 = beta * randn(nx, 1);
        subg2 = th2 + ah * lm2.eqlin + noise2 ...
                + gamma * (thmat2(:,k) - thmat2(:,k-1)); % Subgradient component 2
        subg2 = subg2 / norm(subg2);
        xf2 = Af2 * x02 + bf2 * q2;
        
        % SOLUTION FOR NODE 3 Given Theta
        th3 = thmat3(:,k); % Theta
        [q3,~,~,~,lm3] = quadprog(2*Hvals{3}, cvals{3}, ...
                                  LHS, RHS, Aeqvals{3}, th3-beqvals{3}, ...
                                  [], [], [], options);
        noise3 = beta * randn(nx, 1);
        subg3 = th3 + ah * lm3.eqlin + noise3 ...
                + gamma * (thmat3(:,k) - thmat3(:,k-1)); % Subgradient component 3
        subg3 = subg3 / norm(subg3);
        xf3 = Af3 * x03 + bf3 * q3;
        
        % SOLUTION FOR NODE 4 Given Theta
        th4 = thmat4(:,k); % Theta
        [q4,~,~,~,lm4] = quadprog(2*Hvals{4}, cvals{4}, ...
                                  LHS, RHS, Aeqvals{4}, th4-beqvals{4}, ...
                                  [], [], [], options);
        noise4 = beta * randn(nx, 1);
        subg4 = th4 + ah * lm4.eqlin + noise4 ...
                + gamma * (thmat4(:,k) - thmat4(:,k-1)); % Subgradient component 4
        subg4 = subg4 / norm(subg4);
        xf4 = Af4 * x04 + bf4 * q4;
        
        %Concatenate subgradient
        sub=[subg1,subg2,subg3,subg4];

        %Compute new estimate of theta for each agent (and project)
        thmat1(:,k+1)=project(vert1,sub*Wpow(1,:)');
        thmat2(:,k+1)=project(vert2,sub*Wpow(2,:)');
        thmat3(:,k+1)=project(vert3,sub*Wpow(3,:)');
        thmat4(:,k+1)=project(vert4,sub*Wpow(4,:)');

        %Compute error with respect to optimal centralized solution
        uopt_concat=[q1;q2;q3;q4];
        err = norm(uopt_concat - cent.us) / norm(cent.us);
        err_ev = [err_ev; err];
        dxf=norm(xf1-xf2)+norm(xf1-xf3)+...
            norm(xf1-xf4)+norm(xf2-xf3)+...
            norm(xf2-xf4)+norm(xf3-xf4);
        dxf_ev=[dxf_ev;dxf];
        
        k=k+1;
    end
    dxf_seq{end + 1} = dxf_ev;
    error_sequences{end + 1} = err_ev;
end
%% Final Trajectories
usf=[q1,q2,q3,q4];
plot_agent_trajectories(Phi, Gam, usf, x0f, Tf)
%% Logarithmic Error plot
figure
semilogy(error_sequences{1}, '-*', 'DisplayName','$\phi=1$','LineWidth',2) 
hold on
semilogy(error_sequences{2}, '-^','DisplayName','$\phi=5$','LineWidth',2)
semilogy(error_sequences{3}, '-o','DisplayName','$\phi=10$','LineWidth',2) 
semilogy(error_sequences{4}, '-square','DisplayName','$\phi=20$','LineWidth',2)
grid on
xlabel('$k$','Interpreter','latex');
ylabel('$\frac{||\mathbf{\overline{u}}-\mathbf{\overline{u}}^*||}{||\mathbf{\overline{u}}^*||}$','Interpreter','latex');
title('$\epsilon$ convergence - $\alpha=1/\sqrt{k}$','Interpreter','latex');
set(gca,'FontSize', 18, 'TickLabelInterpreter', 'latex')
legend show

% figure
% semilogy(dxf_seq{1}, '-', 'DisplayName','$\phi=1$','LineWidth',2) 
% hold on
% semilogy(dxf_seq{2}, '-','DisplayName','$\phi=5$','LineWidth',2)
% semilogy(dxf_seq{3}, '-','DisplayName','$\phi=10$','LineWidth',2) 
% semilogy(dxf_seq{4}, '-','DisplayName','$\phi=20$','LineWidth',2)
% grid on
% xlabel('$k$','Interpreter','latex');
% %ylabel('$||\overline{u}_i-\overline{u}_i^*||$','Interpreter','latex');
% title('$\Delta_{x_f}$ convergence - constant $\alpha$','Interpreter','latex');
% set(gca,'FontSize', 18, 'TickLabelInterpreter', 'latex')


%% Functions
function Phi = computePhi(A, Tf)
    % Compute the Phi matrix
    nx = size(A, 1);
    Phi = zeros(nx * Tf, nx);
    for t = 1:Tf
        Phi((t-1)*nx+1:t*nx, :) = A^t;
    end
end

function Gamma = computeGamma(A, B, Tf)
    % Compute the Gamma matrix
    nx = size(A, 1);
    nu = size(B, 2);
    Gamma = zeros(nx * Tf, nu * (Tf-1));
    for t = 1:Tf
        for k = 0:t-1
            Gamma((t-1)*nx+1:t*nx, k*nu+1:(k+1)*nu) = A^(t-1-k) * B;
        end
    end
end

function [Af, bf] = xfMat(A,B,Tf)
%Compute matrices yielding the final state
nx = size(A, 1);
nu = size(B, 2);
Af = A^Tf;
bf = zeros(nx, nu*Tf);
for k=1:Tf
    bf(:,(k-1)*nu+1:(k)*nu)=A^(Tf-k)*B;
end
end

function [H, c] = genCost(Phi,Gamma,x0,nx,nu,Tf)
    H = Gamma'*Gamma + eye(Tf*nu);
    c = (2*x0'*Phi'*Gamma);
end

function [LHS,RHS] = ubound(nu,umax,Tf)
    %Generate left- (LHS) and right-hand side (RHS) of
    %the bound for u
    LHS = kron(eye(nu*Tf), [1; -1]);
    RHS = (umax/Tf) * ones(2*nu*Tf,1);
end

function sel = selecMat(r,nx,na)
    p=zeros(na,1);
    for i=1:na
        p(i)=(i-1)*nx+1;
    end
    %3 constraints in the master problem
    sel.l1=[unitVec(p(1),r),unitVec(p(1)+1,r),...
           unitVec(p(1)+2,r),unitVec(p(1)+3,r)];

    sel.l2=[unitVec(p(2),r),unitVec(p(2)+1,r),...
           unitVec(p(2)+2,r),unitVec(p(2)+3,r)];

    sel.l3=[unitVec(p(3),r),unitVec(p(3)+1,r),...
           unitVec(p(3)+2,r),unitVec(p(3)+3,r)];
    
    sel.l4=[unitVec(p(4),r),unitVec(p(4)+1,r),...
           unitVec(p(4)+2,r),unitVec(p(4)+3,r)];

    % sel.l41=[unitVec(p(4),r),unitVec(p(4)+1,r),...
    %        unitVec(p(4)+2,r),unitVec(p(4)+3,r)];
end

function e = unitVec(i,N)
    e=zeros(N,1);
    e(i)=1;
end

function xfproj=project(M,xf)
    %Check if vector is a coni
    nx=length(xf);
    Aeq = [M'; ones(1,nx)];
    beq = [xf; 1];
    f = zeros(nx,1);
    % Solve the linear program
    options = optimoptions('linprog', 'Display', 'none');
    [~, ~, exitflag] = linprog(f, [],[], Aeq, beq, [], [], options);

    % Check if the solution is feasible
    if exitflag == 1 
        xfproj=xf;
    else
        %Projection matrix
        Mproj=M*pinv(M'*M)*M';
        %Project vector
        xfproj=Mproj*xf;
    end
end

function plot_agent_trajectories(Phi, Gam, usf, x0, Tf)

    % Simulate Trajectories
    x1_seq = Phi.a1*x0(:,1) + Gam.a1*usf(:,1);
    x2_seq = Phi.a2*x0(:,2) + Gam.a2*usf(:,2);
    x3_seq = Phi.a3*x0(:,3) + Gam.a3*usf(:,3);
    x4_seq = Phi.a4*x0(:,4) + Gam.a4*usf(:,4);

    pos_seq1 = zeros(Tf, 4);
    pos_seq2 = zeros(Tf, 4);
    pos_seq3 = zeros(Tf, 4);
    pos_seq4 = zeros(Tf, 4);

    for element_idx = 1:4
        t = 1;
        for j = element_idx:4:(element_idx + 4*(Tf-1))
            pos_seq1(t, element_idx) = x1_seq(j, :);
            pos_seq2(t, element_idx) = x2_seq(j, :);
            pos_seq3(t, element_idx) = x3_seq(j, :);
            pos_seq4(t, element_idx) = x4_seq(j, :);
            t = t + 1;
        end
    end

    % Plot x and x velocity
    figure

    subplot(2, 1, 1)
    plot(pos_seq1(:, 1), '-o', 'LineWidth', 3, 'MarkerSize', 10); 
    hold on
    plot(pos_seq2(:, 1), '--s', 'LineWidth', 3, 'MarkerSize', 10); 
    plot(pos_seq3(:, 1), ':d', 'LineWidth', 3, 'MarkerSize', 10, 'Color','green'); 
    plot(pos_seq4(:, 1), '-.*', 'LineWidth', 3, 'MarkerSize', 10); 
    grid on
    set(gca, 'GridAlpha', 0.35, 'FontSize', 18, 'TickLabelInterpreter', 'latex')
    legend('Agent 1', 'Agent 2', 'Agent 3', 'Agent 4')
    ylabel('position', 'Interpreter', 'latex');
    title('Agent Trajectories - $x$ and $\dot{x}$', 'Interpreter', 'latex');
    
    subplot(2, 1, 2)
    plot(pos_seq1(:, 2), '-o', 'LineWidth', 3, 'MarkerSize', 10); 
    hold on
    plot(pos_seq2(:, 2), '--s', 'LineWidth', 3, 'MarkerSize', 10); 
    plot(pos_seq3(:, 2), ':d', 'LineWidth', 3, 'MarkerSize', 10, 'Color','green'); 
    plot(pos_seq4(:, 2), '-.*', 'LineWidth', 3, 'MarkerSize', 10); 
    grid on
    set(gca, 'GridAlpha', 0.35, 'FontSize', 18, 'TickLabelInterpreter', 'latex')
    xlabel('t', 'Interpreter', 'latex');
    ylabel('velocity', 'Interpreter', 'latex');

    % Plot y and y velocity
    figure

    subplot(2, 1, 1)
    plot(pos_seq1(:, 3), '-o', 'LineWidth', 3, 'MarkerSize', 10); 
    hold on
    plot(pos_seq2(:, 3), '--s', 'LineWidth', 3, 'MarkerSize', 10); 
    plot(pos_seq3(:, 3), ':d', 'LineWidth', 3, 'MarkerSize', 10, 'Color','green'); 
    plot(pos_seq4(:, 3), '-.*', 'LineWidth', 3, 'MarkerSize', 10); 
    grid on
    set(gca, 'GridAlpha', 0.35, 'FontSize', 18, 'TickLabelInterpreter', 'latex')
    legend('Agent 1', 'Agent 2', 'Agent 3', 'Agent 4')
    ylabel('position', 'Interpreter', 'latex');
    title('Agent Trajectories - $y$ and $\dot{y}$', 'Interpreter', 'latex');
    
    subplot(2, 1, 2)
    plot(pos_seq1(:, 4), '-o', 'LineWidth', 3, 'MarkerSize', 10); 
    hold on
    plot(pos_seq2(:, 4), '--s', 'LineWidth', 3, 'MarkerSize', 10); 
    plot(pos_seq3(:, 4), ':d', 'LineWidth', 3, 'MarkerSize', 10, 'Color','green'); 
    plot(pos_seq4(:, 4), '-.*', 'LineWidth', 3, 'MarkerSize', 10); 
    grid on
    set(gca, 'GridAlpha', 0.35, 'FontSize', 18, 'TickLabelInterpreter', 'latex')
    xlabel('t', 'Interpreter', 'latex');
    ylabel('velocity', 'Interpreter', 'latex');
end