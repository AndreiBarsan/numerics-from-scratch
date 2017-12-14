o = sym('o', [3, 1], 'real');
t = sym('t', [3, 1], 'real');
th = sym('th', [1, 1], 'real');

% RAfoo = sym('RA', [3, 3], 'real');
% [RA, foo] = qr(RAfoo);
% RA = simplify(RA);
RA = sym('RA', [3, 3], 'real');
tA = sym('tA', [3, 1], 'real');

P = sym('P', [3, 1], 'real');
Pc = RA' * (P - tA);

% Coposition of SE(3) elements:
% (R1, t1)(R2, t2) = (R1 * R2, R1 * t2 + t1)
% (R, t) (xi_R, xi_t) = (R * xi_R, R * xi_t + t)

% Andrei's goal, as of December 14: Show that
%   g(T exp(xi_hat), Pw) = (R * exp(xi_omega_hat))' (Pw - (R * xi_t + t) )
%                        = Pc + skew(Pc) * xi_omega - xi_t
% Where Pc = R' * (Pw - t)
%       R_xi = exp(xi_omega_hat * th)
% Note that xi_t is NOT nu! In fact, its formula is the pretty ugly one from the
% book, (I - R_xi) * (omega x nu) + omega * omega' * nu * theta) ...


% R = expm(skewsym(o) * th);
% Expression from A Mathematical Introduction to Robo. Manip.
% (If we use the built-in expm, we ignore the fact that the exponent matrix is 
% skew-symmetric, so the results become overly complicated!)
cth = cos(th);
sth = sin(th);
nu = (1 - cth);
R = [ cth + o(1)^2 * nu,       o(1)*o(2)*nu - o(3) * sth, o(1)*o(3)*nu + o(2) * sth; ...
      o(3)*sth + o(1)*o(2)*nu, cth + o(2)^2 * nu,         o(2)*o(3)*nu - o(1) * sth; ...
      o(1)*o(3)*nu - o(2)*sth, o(2)*o(3)*nu + o(1) * sth, cth + o(3)^2 * nu   ];
  
tt = (eye(3) - R) * skewsym(o) * t + o * o' * t * th;

tform = [R, tt; 0 0 0 1];
% tform(1,4), one of the more complicated terms of the final expression,
% was verified via manual calculation and looks OK. This means R itself is
% good.

fprintf('Before simplification:\n');
tform(1, 4)

fprintf('After simplification:\n');
simplify(tform(1, 4))


g = (RA * R)' * (P - (RA * tt + tA));
% It would be nice to see the steps automatically generated. Perhaps Mathematica
% would work better in that use case.
% g_mine = R' * Pc - R' * skewsym(o) * t + skewsym(o) * t - R' * o * o' * t * th;
% g_mine = R' * RA' * P - R' * RA' * (RA * tt + tA);
% g_mine = R' * RA' * P - R' * RA' * RA * tt - R' * RA' * tA;
% at this point, MATLAB starts getting confused, since it does not know RA is an
% orthogonal matrix.
g_start = R' * RA' * P - R' * tt - R' * RA' * tA;
% g_mine  = R' * RA' * P - R' * ((eye(3) - R) * skewsym(o) * t + o * o' * t * th) - R' * RA' * tA;
% g_mine  = R' * RA' * P - R' * (eye(3) - R) * skewsym(o) * t - R' * o * o' * t * th - R' * RA' * tA;
% Reset chain, since MATLAB can't know omega (o) is of norm 1.
g_start  = R' * Pc - R' * skewsym(o) * t + skewsym(o) * t -  R' * o * o' * t * th;
% g_mine  = R' * Pc - R' * skewsym(o) * t + skewsym(o) * t -  R' * o * o' * t * th;

g_mine = Pc + skewsym(Pc) * o - t;

disp('Simplified g_start:');
g_start = simplify(g_start, 'Steps', 10)

disp('Simplified g_mine:');
g_mine = simplify(g_mine, 'Steps', 10)

if g_start == g_mine
  disp('[1] g_start and g_mine are the same!')
else
  disp('[1] g_start and g_mine are NOT the same!')
end
if isequal(g, g_mine)
  disp('[2] g_start and g_mine are the same!')
else
  disp('[2] g_start and g_mine are NOT the same!')
end

disp('Diff of first element:');
simplify(g_start(1) - g_mine(1), 'Steps', 500)

disp('Diff of second element:');
simplify(g_start(2) - g_mine(2))

disp('Diff of third element:');
simplify(g_start(3) - g_mine(3))

disp('Diff vector:');
simplify(g_start - g_mine)