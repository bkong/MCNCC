function gen_resnetfeats_maps(db_ind)
imscale = 0.5;

if nargin<1
  db_ind = 2;
end


[db_attr, db_chunks, dbname] = get_db_attrs('maps', db_ind);

trace_H = 300;
trace_W = 300;
aerial_ims = zeros(trace_H, trace_W, 3, 2194, 'single');
map_ims = zeros(trace_H, trace_W, 3, 2194, 'single');
for i=1:1096
  im = imresize(imread(fullfile('datasets', 'maps', 'train', ...
                                sprintf('%d.jpg', i))), ...
                imscale);
  aerial_ims(:,:,:, i) = im(:, 1:width/2, :);
  map_ims(:,:,:, i)    = im(:, width/2+1:end, :);
end
for i=1:1098
  im = imresize(imread(fullfile('datasets', 'maps', 'val', ...
                                sprintf('%d.jpg', i))), ...
                imscale);
  aerial_ims(:,:,:, 1096+i) = im(:, 1:width/2, :);
  map_ims(:,:,:, 1096+i)    = im(:, width/2+1:end, :);
end

% zero-center the data
mean_im = mean(aerial_ims, 4);
mean_im_pix = mean(mean(mean_im, 1), 2);
aerial_ims = bsxfun(@minus, aerial_ims, mean_im_pix);
mean_im = mean(map_ims, 4);
mean_im_pix = mean(mean(mean_im, 1), 2);
map_ims = bsxfun(@minus, map_ims, mean_im_pix);


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
  net.params(1).value = reshape(single([1 0 0]), 1, 1, 3, 1);
end

for c=1:numel(db_chunks)
  % generate features for aerial images
  data = {'data', aerial_ims(:,:,:, db_chunks{c})};
  all_db_feats = generate_db_CNNfeats(net, data);

  for i=1:size(all_db_feats, 4)
    db_feats = all_db_feats(:,:,:, i);
    save(fullfile('feats', dbname, sprintf('aerial_%04d.mat', db_chunks{c}(i))), ...
         'db_feats', '-v7.3');
  end
  clear all_db_feats

  % generate features for map images
  data = {'data', map_ims(:,:,:, db_chunks{c})};
  all_db_feats = generate_db_CNNfeats(net, data);

  for i=1:size(all_db_feats, 4)
    db_feats = all_db_feats(:,:,:, i);
    save(fullfile('feats', dbname, sprintf('map_%04d.mat', db_chunks{c}(i))), ...
         'db_feats', '-v7.3');
  end
  clear all_db_feats
end
end
