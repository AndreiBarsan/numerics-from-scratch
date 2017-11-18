err_norms = [];
range = 1:3:120;
for size=range

  A = rand(size, size);
  A = tril(A);
  A = A + 0.1 * eye(size);
  
  x_gt = rand(size, 1);

  b = A * x_gt;
%   x_sol = A\b;
  x_sol = linsolve(A, b);

  err_norms = [err_norms, norm(x_gt - x_sol)];
end

plot(range, err_norms);
xlabel("Number of equations");
ylabel("Norm of solution error (log scale)");
set(gca, 'YScale', 'log')
