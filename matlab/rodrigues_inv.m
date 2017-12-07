
for i = 1:10000
  twist = rand(3, 1);
  twist_mag = norm(twist, 2);
  twist_axis = twist / twist_mag;
  rotm = vrrotvec2mat([twist_axis; twist_mag]);
  
  rotm_inv = vrrotvec2mat([twist_axis; -twist_mag]);
  
  res = rotm * rotm_inv;
  if ~all(res - eye(3) < 1e-12)
    error('Found twist whose negative is NOT the inverse.')
  end
end


fprintf('Everything OK!\n');