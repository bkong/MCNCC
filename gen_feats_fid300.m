function gen_feats_fid300(db_ind)
imscale = 0.5;

if nargin<1
  db_ind = 2;
end

[db_attr, ~, dbname] = get_db_attrs('fid300', db_ind);

trace_H = 586;
trace_W = 270;
ims = zeros(trace_H,trace_W,1,1175, 'single');
for i=1:1175
  im = imresize(imread(fullfile('datasets', ...
                                'FID-300', ...
                                'references', ...
                                sprintf('%05d.png', i))), ...
                imscale);
  % only 4 shoes are 1 pixel taller (height=587)
  im = im(1:586,:);

  % manually flip left shoe
  if any(ismember([107 525], i))
    im = fliplr(im);
  end

  % pad to max width
  w = size(im,2);
  im = padarray(im, [0 floor((270-w)/2)], 255, 'pre');
  im = padarray(im, [0 ceil((270-w)/2)], 255, 'post');

  ims(:,:,1,i) = im;
end

% zero-center the data
mean_im = mean(ims, 4);
mean_im_pix = mean(mean(mean_im,1),2);
ims = bsxfun(@minus, ims, mean_im_pix);
ims = repmat(ims, 1,1,3,1);


groups = { ...
[162 390 881]; ...
[661 662 1023]; ...
[24 701]; ...
[25 604]; ...
[35 89]; ...
[45 957]; ...
[87 433]; ...
[115 1075]; ...
[160 1074]; ...
[196 813]; ...
[270 1053]; ...
[278 1064]; ...
[306 828]; ...
[363 930]; ...
[453 455]; ...
[656 788]; ...
[672 687]; ...
[867 1015]; ...
[902 1052]; ...
[906 1041]; ...
[1018 1146]; ...
[1065 1162]; ...
[1156 1157]; ...
[1169 1170]; ...
};

treadids = zeros(1175, 1);
id = 0;
for g=1:numel(groups)
  id = id+1;
  for p=groups{g}
    treadids(p) = id;
  end
end
for p=1:1175
  if treadids(p)==0
    id = id+1;
    treadids(p) = id;
  end
end


% load and modify network
net = dagnn.DagNN();
if db_ind==0
  net.addLayer('identity', dagnn.Conv('size', [1 1 3 1], ...
                                      'stride', 1, ...
                                      'pad', 0, ...
                                      'hasBias', false), ...
               {'data'}, {'raw'}, {'I'});
  net.params(1).value = reshape(single([1 0 0]), 1,1,3,1);
else
  flatnn = load(fullfile('models', db_attr{3}));
  net = net.loadobj(flatnn);
  ind = net.getLayerIndex(db_attr{2});
  net.layers(ind:end) = []; net.rebuild();
end


% generate database
data = {net.vars(1).name, ims};
all_db_feats = generate_db_CNNfeats(net, data);
% generate labels for db
all_db_labels = reshape(treadids, 1,1,1,[]);

feat_idx = numel(net.vars);
feat_dims = size(net.vars(end).value);
rfs = net.getVarReceptiveFields(1);
rfsIm = rfs(end);


mkdir(fullfile('feats', dbname))
db_feats = all_db_feats(:,:,:,1);
db_labels = all_db_labels(:,:,:,1);
save(fullfile('feats', dbname, 'fid300_001.mat'), ...
  'db_feats', 'db_labels', 'feat_dims', ...
  'rfsIm', 'trace_H', 'trace_W', '-v7.3')
for i=2:size(all_db_feats,4)
  db_feats = all_db_feats(:,:,:,i);
  db_labels = all_db_labels(:,:,:,i);
  save(fullfile('feats', dbname, sprintf('fid300_%03d.mat', i)), ...
    'db_feats', 'db_labels', '-v7.3');
end
end
