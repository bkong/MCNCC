function all_scores = weighted_masked_NCC_features(db, tpl, mask, w_cell)
  N = size(db, 4);
  batchSize = 200;
  numBatches = ceil(N/batchSize);

  all_scores = cell(numel(w_cell), 1);
  for m=1:numel(w_cell)
    all_scores{m} = zeros(1,1,1,N, 'single');
  end
  for b=1:numBatches
    inds = (b-1)*batchSize+1:min((b-1)*batchSize+batchSize, N);
    feats = masked_NCC_features(db(:,:,:,inds), tpl, mask);
    for m=1:numel(w_cell)
      all_scores{m}(1,1,1,inds) = ...
        gather(sum(sum( vl_nnconv(feats, w_cell{m}, []), 1),2))./size(feats, 3);
    end
  end
end


function feat = masked_NCC_features(IM, TPL, MASK)
  [H,W,C] = size(TPL);

  assert(size(MASK,1)==size(TPL,1) && size(MASK,2)==size(TPL,2) && ...
         size(MASK,4)==1)

  assert(size(IM,1)==size(TPL,1) && size(IM,2)==size(TPL,2))
  nonzero = sum(MASK(:));
  IM = bsxfun(@times, IM, MASK); % zero out invalid region
  mu = sum(sum(IM, 1),2)./nonzero; % compute mean of valid region
  IM = bsxfun(@minus, IM, mu);
  IM = bsxfun(@times, IM, MASK); % keep invalid region zero
  IM_norm = sum(sum(IM.^2, 1),2);

  TPL = bsxfun(@times, TPL, MASK);
  mu = sum(sum(TPL, 1),2)./nonzero;
  TPL = bsxfun(@minus, TPL, mu);
  TPL = bsxfun(@times, TPL, MASK);
  TPL_norm = sum(sum(TPL.^2, 1),2);

  numer = bsxfun(@times, IM, TPL);
  denom = sqrt( bsxfun(@times, IM_norm, TPL_norm)+1e-5 );
  feat = bsxfun(@rdivide, numer, denom);
  feat = bsxfun(@times, feat, MASK);
end
