% Generate test data for the Python code
test_twists = zeros(10, 3);
test_matrices = zeros(10, 3, 3);
for i = 1:1000
  twist = rand(3, 1);
  twist_mag = norm(twist, 2);
  twist_axis = twist / twist_mag;
  rotm = vrrotvec2mat([twist_axis; twist_mag]);
  
  test_twists(i, :) = twist;
  test_matrices(i, :, :) = rotm;
end

test_twists
test_matrices

save('../data/test/so3_twist.mat', 'test_twists');
save('../data/test/so3_rot.mat', 'test_matrices');
