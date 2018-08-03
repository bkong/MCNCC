function gen_query_cache_israeli(db_ind)
erode_pct = 0.1;

if nargin<1
  db_ind = 2;
end


[db_attr, db_chunks, dbname] = get_db_attrs('israeli', db_ind);


% load and modify network
net = dagnn.DagNN();
if db_ind==0
  net.addLayer('identity', dagnn.Conv('size', [1 1 3 1], ...
                                      'stride', 1, ...
                                      'pad', 0, ...
                                      'hasBias', false), ...
               {'data'}, {'raw'}, {'I'});
  net.params(1).value = reshape(single([1 0 0]), 1, 1, 3, 1);
else
  flatnn = load(fullfile('models', db_attr{3}));
  net = net.loadobj(flatnn);
  ind = net.getLayerIndex(db_attr{2});
  net.layers(ind:end) = []; net.rebuild();
end
load(fullfile('results', 'latent_ims_mean_pix.mat'), 'mean_im_pix')
mean_im_pix = gpuArray(mean_im_pix);



dat = load(fullfile('datasets', 'israeli', 'preprocessed_data.mat'), ...
           'print_ims', 'print_masks');
for p=1:size(dat.print_ims, 4)
  print_ims{p} = crop_im_from_mask(dat.print_ims(:,:,:, p), ...
                                   dat.print_masks(:,:,:, p));
end



load(fullfile('feats', dbname, 'israeli_001.mat'), ...
    'feat_dims', 'rfsIm', 'trace_H', 'trace_W')
im_f2i = feat_2_image(rfsIm);
featLen = feat_dims(3);


radius = max(1, floor(min(feat_dims(1), feat_dims(2))*erode_pct));
se = strel('disk', radius, 0);



% extract and cache ResNet features
angles = -20:4:20;
for p=1:numel(print_ims)
  p_im = single(print_ims{p});
  cache_fname = fullfile('feats', dbname, ...
                         sprintf('israeli_cache_%s.mat', ...
                                 hashMat([size(p_im) hashMat(p_im(:))])));
  if exist(cache_fname, 'file'), continue, end

  [p_H, p_W, p_C] = size(p_im);
  % pad latent print so that for every pixel location (of the original latent print)
  % we can extract an image of the same size as the test impressions
  pad_H = trace_H-p_H; pad_W = trace_W-p_W;
  p_im_padded = padarray(p_im, [pad_H pad_W], 255, 'both');
  p_mask_padded = padarray(ones(p_H, p_W, 'logical'), [pad_H pad_W], 0, 'both');

  % generate rotation images
  p_im_padded_r_stack = zeros(size(p_im_padded, 1), size(p_im_padded, 2), p_C, numel(angles), ...
                              'like', p_im_padded);
  p_mask_padded_r_stack = zeros(size(p_mask_padded, 1), size(p_mask_padded, 2), 1, numel(angles), ...
                                'like', p_mask_padded);
  for a=1:numel(angles)
    p_im_padded_r_stack(:,:,:, a) = imrotate(p_im_padded, angles(a), 'bicubic', 'crop');
    p_mask_padded_r_stack(:,:,:, a) = imrotate(p_mask_padded, angles(a), 'nearest', 'crop');
  end

  % generate features for translations and rotations
  offsets_y = 0;
  if pad_H>1
    offsets_y = [0 2];
  end
  offsets_x = 0;
  if pad_W>1
    offsets_x = [0 2];
  end
  p_r_feats = cell(numel(offsets_y), numel(offsets_x));
  p_r_feat_masks = cell(numel(offsets_y), numel(offsets_x));
  for ox_ind=1:numel(offsets_x), for oy_ind=1:numel(offsets_y)
    p_im_padded_r_stack_patch = p_im_padded_r_stack(offsets_y(oy_ind)+1:end, ...
                                                    offsets_x(ox_ind)+1:end, :,:);
    p_r_feats{oy_ind,ox_ind} = generate_db_CNNfeats(net, ...
      {net.vars(1).name, bsxfun(@minus, p_im_padded_r_stack_patch, mean_im_pix)});

    p_r_masks = cell(size(p_r_feats{oy_ind,ox_ind}, 1)-feat_dims(1)+1, ...
                     size(p_r_feats{oy_ind,ox_ind}, 2)-feat_dims(2)+1);
    % shifting by 1 in the feature space = shifting by 4px in the image space
    for j=1:size(p_r_feats{oy_ind,ox_ind}, 2)-feat_dims(2)+1
    for i=1:size(p_r_feats{oy_ind,ox_ind}, 1)-feat_dims(1)+1
      pix_i = offsets_y(oy_ind)+(i-1)*4+1; pix_j = offsets_x(ox_ind)+(j-1)*4+1;
      % skip features outside the image
      if pix_i+trace_H-1>size(p_mask_padded_r_stack, 1) || ...
         pix_j+trace_W-1>size(p_mask_padded_r_stack, 2),
        continue
      end

      % just compute the wrapped mask everytime to simplify code logic
      p_ijr_mask = p_mask_padded_r_stack(pix_i:pix_i+trace_H-1, pix_j:pix_j+trace_W-1, 1, :);
      p_ijr_feat_mask = warp_masks(p_ijr_mask, im_f2i, feat_dims, db_ind);
      % erode masks
      p_ijr_feat_mask = padarray(p_ijr_feat_mask, [radius radius], 0);
      p_ijr_feat_mask = imerode(p_ijr_feat_mask, se, 'same');
      p_ijr_feat_mask = p_ijr_feat_mask(radius+1:end-radius, radius+1:end-radius, :,:);

      assert(all(squeeze(sum(sum( p_ijr_feat_mask, 1), 2))>0))
      p_r_masks{i,j} = p_ijr_feat_mask;
    end
    end
    p_r_feat_masks{oy_ind,ox_ind} = p_r_masks;
  end, end

  % cache features
  save(cache_fname, 'p_r_feats', 'p_r_feat_masks', 'p_im_padded_r_stack')
end

end
