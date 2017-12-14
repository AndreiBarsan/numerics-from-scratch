function [ res ] = skewsym( vv )
  res = [ ...
    0 -vv(3) vv(2) ; 
    vv(3) 0 -vv(1) ;
    -vv(2) vv(1) 0 ];
end

