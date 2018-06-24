function no_search_eval_maps(db_ind, disttype)

[db_attr, db_chunks, dbname] = get_db_attrs('maps', db_ind);

num_db = max(cat(2, db_chunks{:}));
similarity = zeros(num_db, 'single');
for c=1:numel(db_chunks)
  % load database
  for i=1:numel(db_chunks{c})
    data = load(sprintf('feats/%s/map_%04d.mat', dbname, db_chunks{c}(i)));
    if i==1
      db_feats = zeros(size(data.db_feats,1), ...
                       size(data.db_feats,2), ...
                       size(data.db_feats,3), ...
                       numel(db_chunks{c}), 'single');
    end
    db_feats(:,:,:, i) = data.db_feats;
  end
  db_feats = gpuArray(db_feats);

  for q=1:num_db
    query = load(sprintf('feats/%s/aerial_%04d.mat', dbname, q));

    scores = gather(compute_db_scores(db_feats, ...
                                      gpuArray(query.db_feats), ...
                                      disttype));
    similarity(q, db_chunks{c}) = reshape(scores, 1, []);
  end
end

if ~exist(fullfile('results', dbname), 'dir')
  mkdir(fullfile('results', dbname))
end
save(fullfile('results', dbname, sprintf('maps_%s_similarity.mat', disttype)), ...
     'similarity')

end
