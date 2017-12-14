syms x y z t

z = y * 2.5;
t = z / 2.5;

a = exp(x) * exp(y);
b = exp(y + x);

disp('Checking equality:');
a
b
if a == b
  disp('== EQUAL!')
else
  disp('== NOT EQUAL!')
end

a_sim = simplify(a)
b_sim = simplify(b)

if a_sim == b_sim
  disp('== EQUAL')
else
  disp('== NOT EQUAL!')
end