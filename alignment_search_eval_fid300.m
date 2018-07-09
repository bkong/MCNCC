function alignment_search_eval_fid300(p_inds, db_ind)
% seems faster to run database against latent prints, potentially fewer
% translations to look over (especially when latent prints are MUCH smaller than
% test impressions)
imscale = 0.5;
erode_pct = 0.1;

if nargin<2
  db_ind = 2;
end


[db_attr, db_chunks, dbname] = get_db_attrs('basel', db_ind);


% load and modify network
flatnn = load(fullfile('models', 'imagenet-resnet-50-dag.mat'));
net = dagnn.DagNN();
net = net.loadobj(flatnn);
ind = net.getLayerIndex(db_attr{2});
net.layers(ind:end) = []; net.rebuild();
if db_ind==0
  net.addLayer('identity', dagnn.Conv('size', [1 1 3 1], ...
                                      'stride', 1, ...
                                      'pad', 0, ...
                                      'hasBias', false), ...
               {'data'}, {'raw'}, {'I'});
  net.params(1).value = reshape(single([1 0 0]), 1,1,3,1);
end
load(fullfile('results', 'latent_ims_mean_pix.mat'), 'mean_im_pix')
mean_im_pix = mean_im_pix; % stupid MATLAB transparency



% load database chunk
db_chunk_inds = db_chunks{1};
load(fullfile('feats', dbname, 'fid300_001.mat'), ...
  'db_feats', 'feat_dims', 'rfsIm', 'trace_H', 'trace_W')
feat_dims = feat_dims; % stupid MATLAB transparency
db_feats = zeros(size(db_feats,1), size(db_feats,2), size(db_feats,3), ...
  numel(db_chunk_inds), 'like', db_feats);
for i=1:numel(db_chunk_inds)
  dat = load(fullfile('feats', dbname, sprintf('fid300_%03d.mat', db_chunk_inds(i))));
  db_feats(:,:,:,i) = dat.db_feats;
end
im_f2i = feat_2_image(rfsIm);

radius = max(1, floor(min(feat_dims(1), feat_dims(2))*erode_pct));
se = strel('disk', radius, 0);



ones_w = gpuArray.ones(1, 1, feat_dims(3), 'single');



db_feats = gpuArray(db_feats);
for p=reshape(p_inds, 1, [])
  fname = fullfile('results', dbname, sprintf('fid300_alignment_search_ones_res_%04d.mat', p));
  if exist(fname, 'file'), continue, end
  lock_fname = [fname,'.lock'];
  if exist(lock_fname, 'file'), continue, end
  fid = fopen(lock_fname, 'w');
  fprintf('p=%d: ', p),tic

  p_im = imresize(imread(fullfile('datasets', 'FID-300', 'tracks_cropped', sprintf('%05d.jpg', p))), ...
                  imscale);
  [p_H,p_W,p_C] = size(p_im);
  % fix latent prints are bigger than the test impressions
  if p_H>p_W
    if p_H>trace_H
      p_im = imresize(p_im, [trace_H NaN]);
    end
  else
    if p_W>trace_W
      p_im = imresize(p_im, [NaN trace_W]);
    end
  end
  p_im = bsxfun(@minus, single(p_im), mean_im_pix);
  [p_H, p_W, p_C] = size(p_im);

  % pad latent print so that for every pixel location (of the original latent print)
  % we can extract an image of the same size as the test impressions
  pad_H = trace_H-p_H; pad_W = trace_W-p_W;
  p_im_padded = padarray(p_im, [pad_H pad_W], 255, 'both');
  p_mask_padded = padarray(ones(p_H, p_W, 'logical'), [pad_H pad_W], 0, 'both');

  cnt = 0;
  eraseStr = '';
  angles = -20:4:20;
  transx = 1:2:pad_W+1;
  transy = 1:2:pad_H+1;
  scores_ones = zeros(numel(transy), ...
                      numel(transx), ...
                      numel(angles), ...
                      size(db_feats,4), 'single');
  for r=1:numel(angles)
    % rotate image and mask
    p_im_padded_r = imrotate(p_im_padded, angles(r), 'bicubic', 'crop');
    p_mask_padded_r = imrotate(p_mask_padded, angles(r), 'nearest', 'crop');

    % NOTE: this works for res2bx (db_ind=2),
    % need to double-check when using other ResNet features
    offsets_y = 0;
    if pad_H>1
      offsets_y = [0 2];
    end
    offsets_x = 0;
    if pad_W>1
      offsets_x = [0 2];
    end
    for offsetx=offsets_x, for offsety=offsets_y
      p_r_feat = generate_db_CNNfeats_gpu(net, {'data', p_im_padded_r(offsety+1:end, offsetx+1:end, :)});

      % shifting by 1 in the feature space = shifting by 4px in the image space
      for j=1:size(p_r_feat, 2)-feat_dims(2)+1
      for i=1:size(p_r_feat, 1)-feat_dims(1)+1
        msg = sprintf('%d/%d ', cnt, numel(angles)*ceil(pad_H/2+0.5)*ceil(pad_W/2+0.5));
        if mod(cnt, 10)==0
          fprintf([eraseStr, msg]);
          eraseStr = repmat(sprintf('\b'), 1, length(msg));
        end
        fprintf(fid, [msg,'\n']);

        pix_i = offsety+(i-1)*4+1; pix_j = offsetx+(j-1)*4+1;
        % skip features outside the image
        if pix_i+trace_H-1>size(p_mask_padded_r, 1) || ...
           pix_j+trace_W-1>size(p_mask_padded_r, 2),
          continue
        end

        p_ijr_feat = p_r_feat(i:i+feat_dims(1)-1, j:j+feat_dims(2)-1, :);
        % just compute the wrapped mask everytime to simplify code logic
        p_mask_ijr = p_mask_padded_r(pix_i:pix_i+trace_H-1, pix_j:pix_j+trace_W-1);
        p_ijr_feat_mask = warp_masks(p_mask_ijr, im_f2i, feat_dims, db_ind);
        % erode masks
        p_ijr_feat_mask = padarray(p_ijr_feat_mask, [radius radius], 0);
        p_ijr_feat_mask = imerode(p_ijr_feat_mask, se, 'same');
        p_ijr_feat_mask = p_ijr_feat_mask(radius+1:end-radius, radius+1:end-radius, :,:);
        p_ijr_feat_mask = gpuArray(p_ijr_feat_mask);

        % find the argmax score of database entries
        scores_cell = weighted_masked_NCC_features(db_feats, p_ijr_feat, p_ijr_feat_mask, ...
                                                   {ones_w});
        scores_ones(pix_i/2+0.5, pix_j/2+0.5,r, :) = scores_cell{1};
        cnt = cnt+1;
      end
      end
    end, end
  end
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

end
