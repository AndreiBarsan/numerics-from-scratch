% Finds the root of the function 'f' using the Newton-Raphson method.
%
% Set 'f' and 'f_prime' to the function whose root you want to find and its
% derivative (yes, you have to do the math to find its derivative). If you're
% feeling lazy, just use automatic differentiation (Ceres, PyTorch, and JAX are 
% all great examples). You can probably do autodiff in MATLAB but it's likely
% a pain in the butt, lol.
%

fprintf('Newton-Raphson example:\n');

% Basic arguments
x_0 =           15.0;                             % Initial guess
f =             @f_example_from_wikipedia;        % Function to optimize
f_prime =       @f_prime_example_from_wikipedia;  % Analytic derivative of f
eps_threshold = 1e-5;                             % Convergence threshold
max_iters =     100;                              % Maximum number of iterations


% Main loop
x = x_0
for i = 1:max_iters
  x_new = x - f(x) / f_prime(x);
  if abs(x_new - x) < eps_threshold
    break;
  end
  x = x_new;
end


fprintf('Found root:  %12.4f in %d step(s)\n', x, i)
fprintf('Final value: %12.4f = f(%.4f)\n', f(x), x)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function Zoo

function y = f_quad(x)
  y = x * 2 + 3;
end

function y = f_prime_quad(x)
  y = 2;
end

function y = f_example_from_wikipedia(x)
  y = cos(x) - x ^ 3;
end

function y = f_prime_example_from_wikipedia(x)
  y = -sin(x) - 3 * (x ^ 2);
end


