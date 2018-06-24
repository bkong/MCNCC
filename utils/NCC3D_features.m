function feat = NCC3D_features(IM, TPL)
  [H,W,C] = size(TPL);

  if size(IM,1)==H && size(IM,2)==W
    mu = sum(sum(sum(IM, 1),2),3)./(H*W*C); % compute mean
    IM = bsxfun(@minus, IM, mu);
    IM_norm = sum(sum(sum(IM.^2, 1),2),3);

    mu = sum(sum(sum(TPL, 1),2),3)./(H*W*C);
    TPL = bsxfun(@minus, TPL, mu);
    TPL_norm = sum(sum(sum(TPL.^2, 1),2),3);

    numer = bsxfun(@times, IM, TPL);
    denom = sqrt( bsxfun(@times, IM_norm, TPL_norm)+1e-5 );
    feat = bsxfun(@rdivide, numer, denom);
  else
    error('IM and TPL must have the same H,W,C')
  end
end
