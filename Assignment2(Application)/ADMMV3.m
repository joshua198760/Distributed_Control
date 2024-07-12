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
%% ADMM Consensus
z = zeros(4,1);%Final State
%Initialize Lambdas
l1 = zeros(4,1); l2 = zeros(4,1); 
l3 = zeros(4,1); l4 = zeros(4,1);
%Error sequence
err_ev=[];error_sequences={};
%Regularization term
rho_vals=linspace(1,40,80);
Knum=[];
%Iterations Params
N=320;
for rho=rho_vals
    rho
    k = 1;
    z = zeros(4,1);%Final State
    %Initialize Lambdas
    l1 = zeros(4,1); l2 = zeros(4,1); 
    l3 = zeros(4,1); l4 = zeros(4,1);
    %Error sequence
    err=inf;
    err_ev=[];
    while err>1e-3
        opts = optimoptions('quadprog', 'Display', 'off');
        %Node 1
        H1d = (H1 + rho/2*bf1'*bf1) + (H1 + rho/2*bf1'*bf1)';
        u1 = quadprog(H1d,(c1 + l1'*bf1 + rho*(Af1*x01 - z)'*bf1),LHS,RHS,[],[],[],[],[],opts);
        %Node 2
        H2d = (H2 + rho/2*bf2'*bf2) + (H2 + rho/2*bf2'*bf2)';
        u2 = quadprog(H2d,(c2 + l2'*bf2 + rho*(Af2*x02 - z)'*bf2),LHS,RHS,[],[],[],[],[],opts);
        %Node 3
        H3d = (H3 + rho/2*bf3'*bf3) + (H3 + rho/2*bf3'*bf3)';
        u3 = quadprog(H3d,(c3 + l3'*bf3 + rho*(Af3*x03 - z)'*bf3),LHS,RHS,[],[],[],[],[],opts);
        %Node 4
        H4d = (H4 + rho/2*bf4'*bf4) + (H4 + rho/2*bf4'*bf4)';
        u4 = quadprog(H4d,(c4 + l4'*bf4 + rho*(Af4*x04 - z)'*bf4),LHS,RHS,[],[],[],[],[],opts);
    
        %Update final state
        z = ((l1/rho+bf1*u1+Af1*x01)+(l2/rho + bf2*u2 + Af2*x02)+ ...
            (l3/rho + bf3*u3 + Af3*x03) + (l4/rho + bf4*u4 + Af4*x04))/na;
        
        %Update dual variables
        l1 = l1+rho*(bf1*u1 + Af1*x01 - z);
        l2 = l2 + rho*(bf2*u2 + Af2*x02 - z);
        l3 = l3 + rho*(bf3*u3 + Af3*x03 - z);
        l4 = l4 + rho*(bf4*u4 + Af4*x04 - z);
    
        %Append to error sequence
        uopt_concat = [u1;u2;u3;u4];
        err = norm(uopt_concat - cent.us) / norm(cent.us);
        err_ev = [err_ev; err];

        if err <=1e-3
            disp('min err reached')
            Knum=[Knum;k];
        end
        k=k+1;
    end
    error_sequences{end + 1} = err_ev;
end

%% Plot rho vs k
figure
plot(rho_vals,Knum,'-^','LineWidth',2);
xlabel('$\rho$','Interpreter','latex');
ylabel('Number of Iterations','Interpreter','latex');
title('$\rho$ vs. $k$ to reach $\epsilon\leq \epsilon_{min}$', 'Interpreter','latex')
set(gca,'FontSize', 18, 'TickLabelInterpreter', 'latex');
grid on
%% Final Trajectories
usf=[u1,u2,u3,u4];
plot_agent_trajectories(Phi, Gam, usf, x0f, Tf)
%% Logarithmic Error Plot
figure
semilogy(error_sequences{1}, '-', 'DisplayName','$\rho=1$','LineWidth',2) 
hold on
semilogy(error_sequences{2}, '-','DisplayName','$\rho=5$','LineWidth',2)
semilogy(error_sequences{3}, '-','DisplayName','$\rho=10$','LineWidth',2) 
semilogy(error_sequences{4}, '-','DisplayName','$\rho=20$','LineWidth',2)
grid on
xlabel('$k$','Interpreter','latex');
ylabel('$\frac{||\mathbf{\overline{u}}-\mathbf{\overline{u}}^*||}{||\mathbf{\overline{u}}^*||}$','Interpreter','latex');
title('$\epsilon$ convergence - constant $\alpha$','Interpreter','latex');
set(gca,'FontSize', 18, 'TickLabelInterpreter', 'latex')
legend show

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