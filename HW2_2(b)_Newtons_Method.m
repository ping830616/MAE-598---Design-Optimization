clc
clear
format long

% Function Definition (Enter your Function here):
syms X Y;
f = (2-2*X-3*Y)^2+X^2+(Y-1)^2;

% Initial Guess (Choose Initial Guesses):
x(1) = 3;
y(1) = -4;
e = 10^(-8); % Convergence Criteria
i = 1; % Iteration Counter

% Gradient and Hessian:
df_dx = diff(f, X);
df_dy = diff(f, Y);

J = [subs(df_dx,[X,Y], [x(1),y(1)]) subs(df_dy, [X,Y], [x(1),y(1)])]; % Gradient

ddf_ddx = diff(df_dx,X);
ddf_ddy = diff(df_dy,Y);
ddf_dxdy = diff(df_dx,Y);
ddf_ddx_1 = subs(ddf_ddx, [X,Y], [x(1),y(1)]);
ddf_ddy_1 = subs(ddf_ddy, [X,Y], [x(1),y(1)]);
ddf_dxdy_1 = subs(ddf_dxdy, [X,Y], [x(1),y(1)]);

H = [ddf_ddx_1, ddf_dxdy_1; ddf_dxdy_1, ddf_ddy_1]; % Hessian

S = inv(H); % Search Direction

% Optimization Condition
while norm(J) > e
    I = [x(i),y(i)]';
    x(i+1) = I(1)-S(1,:)*J';
    y(i+1) = I(2)-S(2,:)*J';
    i = i+1;

    J = [subs(df_dx,[X,Y], [x(i),y(i)]) subs(df_dy, [X,Y], [x(i),y(i)])]; % Updated Jacobian

    ddf_ddx_1 = subs(ddf_ddx, [X,Y], [x(i),y(i)]);
    ddf_ddy_1 = subs(ddf_ddy, [X,Y], [x(i),y(i)]);
    ddf_dxdy_1 = subs(ddf_dxdy, [X,Y], [x(i),y(i)]);

    H = [ddf_ddx_1, ddf_dxdy_1; ddf_dxdy_1, ddf_ddy_1]; % Updated Hessian

    S = inv(H); % New Search Direction
end

% Result Table:
Iter = 1:i;
X_coordinate = x';
Y_coordinate = y';
Iterations = Iter';
T = table(Iterations,X_coordinate,Y_coordinate);

% Plots:
fcontour(f, 'Fill', 'On');
hold on;
loglog(x,y,'*-r');


% Output:
fprintf('Initial objective function value: %d\n\n',subs(f,[X,Y], [x(1),y(1)]));

if (norm(J) < e)
    fprintf('Minimum finally reached! \n\n');
end

fprintf('Number of iterations for convergence: %d\n\n', i);
fprintf('The minimum: [%d,%d]\n\n', x(i), y(i));
disp(T)
