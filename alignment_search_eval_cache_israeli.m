function alignment_search_eval_cache_israeli(gpu_id, p_inds, db_ind, use_mask)
% seems faster to run database against latent prints, potentially fewer
% translations to look over (especially when latent prints are MUCH smaller than
% test impressions)
gpuDevice(eval(gpu_id));
p_inds = eval(p_inds);
if nargin<2
  db_ind = 2;
else
  db_ind = eval(db_ind);
end
if nargin<3
  use_mask = true;
else
  use_mask = eval(use_mask);
end


[db_attr, db_chunks, dbname] = get_db_attrs('israeli', db_ind);



% build database chunk
db_chunk_inds = db_chunks{1};
load(fullfile('feats', dbname, 'israeli_001.mat'), ...
  'db_feats', 'feat_dims', 'trace_H', 'trace_W')
feat_dims = feat_dims; % stupid MATLAB transparency
db_feats = zeros(size(db_feats, 1), size(db_feats, 2), size(db_feats, 3), ...
  numel(db_chunk_inds), 'like', db_feats);
for i=1:numel(db_chunk_inds)
  dat = load(fullfile('feats', dbname, sprintf('israeli_%03d.mat', db_chunk_inds(i))));
  db_feats(:,:,:, i) = dat.db_feats;
end



ones_w = gpuArray.ones(1, 1, feat_dims(3), 'single');



dat = load(fullfile('datasets', 'israeli', 'preprocessed_data.mat'), ...
           'print_ims', 'print_masks');
for p=1:size(dat.print_ims, 4)
  print_ims{p} = crop_im_from_mask(dat.print_ims(:,:,:, p), ...
                                   dat.print_masks(:,:,:, p));
end



db_feats = gpuArray(db_feats);
mkdir(fullfile('results', dbname))
for p=reshape(p_inds, 1, [])
  if use_mask
    fname = fullfile('results', dbname, sprintf('israeli_alignment_search_ones_res_%04d.mat', p));
  else
    fname = fullfile('results', dbname, sprintf('israeli_alignment_search_nomask_ones_res_%04d.mat', p));
  end
  if exist(fname, 'file'), continue, end
  lock_fname = [fname, '.lock'];
  if exist(lock_fname, 'file'), continue, end
  fid = fopen(lock_fname, 'w');
  fprintf('p=%d: ', p),tic

  p_im = single(print_ims{p});
  [p_H, p_W, p_C] = size(p_im);

  % pad latent print so that for every pixel location (of the original latent print)
  % we can extract an image of the same size as the test impressions
  pad_H = trace_H-p_H; pad_W = trace_W-p_W;
  [p_im, p_r_feat_stacks, p_r_feat_mask_stacks] = ...
    get_cached_data_pt(cache_path, p_im);

  cnt = 0;
  eraseStr = '';
  angles = -20:4:20;
  transx = 1:2:pad_W+1;
  transy = 1:2:pad_H+1;
  scores_ones = zeros(numel(transy), ...
                      numel(transx), ...
                      numel(angles), ...
                      size(db_feats, 4), 'single');
  offsets_x = [0 2]; offsets_y = [0 2];
  for ox_ind=1:size(p_r_feat_stacks, 2), for oy_ind=1:size(p_r_feat_stacks, 1)
    p_r_feat = gpuArray(p_r_feat_stacks{oy_ind, ox_ind});

    % shifting by 1 in the feature space = shifting by 4px in the image space
    for j=1:size(p_r_feat, 2)-feat_dims(2)+1
    for i=1:size(p_r_feat, 1)-feat_dims(1)+1
      msg = sprintf('%d/%d ', cnt, ceil(pad_H/2+0.5)*ceil(pad_W/2+0.5));
      if mod(cnt, 1)==0
        fprintf([eraseStr, msg]);
        eraseStr = repmat(sprintf('\b'), 1, length(msg));
      end
      fprintf(fid, [msg, '\n']);

      pix_i = offsets_y(oy_ind)+(i-1)*4+1; pix_j = offsets_x(ox_ind)+(j-1)*4+1;
      % skip features outside the image
      if pix_i+trace_H-1>p_H+pad_H*2 || pix_j+trace_W-1>p_W+pad_W*2
        continue
      end

      p_ijr_feat = p_r_feat(i:i+feat_dims(1)-1, j:j+feat_dims(2)-1, :,:);
      p_ijr_feat_mask = gpuArray(p_r_feat_mask_stacks{oy_ind, ox_ind}{i, j});
      assert(size(p_ijr_feat_mask, 3)==1)
      assert(gather(all(squeeze(sum(sum(p_ijr_feat_mask, 1), 2))>0)))

      % compute score for database entries
      for db_i=1:size(db_feats,4)
        scores_cell = weighted_masked_NCC_features(db_feats(:,:,:, db_i), ...
                                                   p_ijr_feat, ...
                                                   p_ijr_feat_mask, ...
                                                   {ones_w});
        scores_ones(pix_i/2+0.5, pix_j/2+0.5, :, db_i) = scores_cell{1};
      end
      cnt = cnt+1;
    end
    end
  end, end
  minsONES = max(max(max(scores_ones, [], 1), [], 2), [], 3);
  locaONES = bsxfun(@eq, scores_ones, minsONES);

  % save results
  save_results(fname, struct('scores_ones', scores_ones, ...
                             'minsONES', minsONES, ...
                             'locaONES', locaONES));
  % remove lockfile
  fclose(fid);
  delete(lock_fname);

  toc
end
