function feat = NCC_features(IM, TPL)
  [H,W,C] = size(TPL);

  if size(IM,1)==H && size(IM,2)==W
    mu = sum(sum(IM, 1),2)./(H*W); % compute mean
    IM = bsxfun(@minus, IM, mu);
    IM_norm = sum(sum(IM.^2, 1),2);

    mu = sum(sum(TPL, 1),2)./(H*W);
    TPL = bsxfun(@minus, TPL, mu);
    TPL_norm = sum(sum(TPL.^2, 1),2);

    numer = bsxfun(@times, IM, TPL);
    denom = sqrt( bsxfun(@times, IM_norm, TPL_norm)+1e-5 );
    feat = bsxfun(@rdivide, numer, denom);
  else
    ncc = normxcorr2e(TPL(:,:,1), IM(:,:,1), 'same');

    feat = zeros(size(ncc,1), size(ncc,2), C, 'like', IM);
    feat(:,:,1) = ncc;
    for c = 2:C
      if numel(unique(reshape(TPL(:,:,c),[],1)))>1
        feat(:,:,c) = normxcorr2e(TPL(:,:,c), IM(:,:,c), 'same');
      end
    end
  end
end
