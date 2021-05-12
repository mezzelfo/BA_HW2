clear all
close all
clc

NITEM = 2;
MAXINV = 4+1; %0 to 4
EPS = 1e-5;
GAMMA = 0.99;

NSTATES = MAXINV^NITEM;

PI = rand(NSTATES,NITEM,NSTATES); %TO BE CONSTRUCTED
F = rand(NSTATES,NITEM);

V = zeros(NSTATES,1);

k = 0;
stop = 0;
while ~stop
    Vnew = zeros('like',V);
    for s = 1:NSTATES
        max(F(s)+GAMMA.*PI(s)*V,[],2)
    end
    stop = 1;
end

function r = immediate_reward(state, action)
    r = 0;
end

function p = pi(idx_from, action, idx_to)
    state_from = ind2sub(MAXINV*ones(NITEM,1),idx_from)-1;
    state_to = ind2sub(MAXINV*ones(NITEM,1),idx_to)-1;
    gamma = state_from - state_to;
    gamma(action) = gamma(action) + PRODUCTION(action);
    
end