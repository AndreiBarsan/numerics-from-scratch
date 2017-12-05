R = sym('R', [3, 3]);
X = sym('X', [3, 1]);
t = sym('t', [3, 1]);
delta = sym('delta', [3, 1]);

P = R * (X + delta) + t;
p = -P(1:2) / P(3);

fprintf('Jacobian of transform\n');
jacobian(P, delta)

fprintf('Jacobian of transform and projection\n');
% TODO(andrei): Can we simplify the resulting expression?
jacobian(p, delta)

skewsym = @(vv) [0 -vv(3) vv(2) ; vv(3) 0 -vv(1) ; -vv(2) vv(1) 0 ];

Q = sym('Q', [3, 1])
Qx = skewsym(Q)
nu = sym('nu', [3, 1]);
w = sym('w', [3, 1]);
syms f;

h1 = f * (Q + Qx * w + nu)
jacobian(h1, w)
jacobian(h1, nu)

rot = expm(skewsym([0, 1, 0]'))
% subs(rot, w, [0, 1, 0]')

vrrotvec2mat([0, 1, 0, 1]')

