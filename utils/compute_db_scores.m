function scores = compute_db_scores(db, query, disttype)
assert(size(db,3)==size(query,3))

N = size(db, 4);
if strcmp(disttype, 'mcncc')
  tmp = sum(sum(sum(NCC_features(db(:,:,:,1), query), 3),2),1);
  scores = zeros(1,1,1, size(db,4), 'like', db);
  scores(:,:,:,1) = tmp;
  for i=2:size(db,4)
    scores(:,:,:,i) = sum(sum(sum(NCC_features(db(:,:,:,i), query), 3),2),1);
  end
  scores = scores./( size(query,1)*size(query,2)*size(query,3) );

elseif strcmp(disttype, '3dncc')
  tmp = sum(sum(sum(NCC3D_features(db(:,:,:,1), query), 3),2),1);
  scores = zeros(1,1,1, size(db,4), 'like', db);
  scores(:,:,:,1) = tmp;
  for i=2:size(db,4)
    scores(:,:,:,i) = sum(sum(sum(NCC3D_features(db(:,:,:,i), query), 3),2),1);
  end
  scores = scores./( size(query,1)*size(query,2)*size(query,3) );

elseif strcmp(disttype, 'cosine')
  db = reshape(db, [], N);
  db = bsxfun(@rdivide, db, sqrt(sum(db.*db, 1)+1e-5));
  query = reshape(query, 1, []) ./ sqrt( (query(:)'*query(:))+1e-5 );
  scores = query*db;

elseif strcmp(disttype, 'euclidean')
  db = reshape(db, [], N);
  query = reshape(query, 1, []);
  scores = -sqrt( query*query(:) - query*db + sum(db.*db, 1) );

else
  error('Unknown disttype!')
end
scores = reshape(scores, 1,1,1,N);

end
