% g(T * exp(xi), X, f) = P + P \cross xi(0:3) - xi(3:6)
%   where
% P = R' (X - t)
%
% dg/dxi = ?

R = sym('R', [3, 3], 'real');
X = sym('X', [3, 1], 'real');
t = sym('t', [3, 1], 'real');
xi = sym('xi', [6, 1], 'real');

T = [R, t; 0 0 0 1];
P = R' * (X - t);

skewsym = @(vv) [0 -vv(3) vv(2) ; vv(3) 0 -vv(1) ; -vv(2) vv(1) 0 ];

size(P)
size(xi(1:3))
size(skewsym(P))


g = P + skewsym(P) * xi(1:3) - xi(4:6)

syms x y z
Jg = [x 0 0 y 0 0 z 0 0 1 0 0; ...
      0 x 0 0 y 0 0 z 0 0 1 0; ...
      0 0 x 0 0 y 0 0 z 0 0 1];
JG = [0 0 0 0 0 0; ...
      0 0 0 0 0 1; ...
      0 0 0 0 -1 0; ...
      0 0 0 0 0 -1; ...
      0 0 0 0 0 0; ...
      0 0 0 1 0 0; ...
      0 0 0 0 1 0; ...
      0 0 0 -1 0 0; ...
      0 0 0 0 0 0; ...
      1 0 0 0 0 0; ...
      0 1 0 0 0 0; ...
      0 0 1 0 0 0];
Jg
JG
Jg * JG


