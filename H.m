function h = H(p)
% highly personalized
p(p==0) = [];
h = -sum(p.*log2(p));
end
