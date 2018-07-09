function gen_resnetfeats_israeli(db_ind)
if nargin<1
  db_ind = 2;
end

[db_attr, ~, dbname] = get_db_attrs('israeli', db_ind);

load(fullfile('datasets', 'israeli', 'preprocessed_data.mat'), ...
     'trace_ims', 'trace_treadids')
trace_H = size(trace_ims, 1); trace_W = size(trace_ims, 2);
% zero-center the data
trace_ims = single(trace_ims);
mean_im = mean(trace_ims, 4);
mean_im_pix = mean(mean(mean_im, 1), 2);
trace_ims = bsxfun(@minus, trace_ims, mean_im_pix);


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


% generate database
data = {'data', trace_ims};
all_db_feats = generate_db_CNNfeats(net, data);
% generate labels for db
all_db_labels = reshape(trace_treadids, 1, 1, 1, []);

feat_idx = numel(net.vars);
feat_dims = size(net.vars(end).value);
rfs = net.getVarReceptiveFields(1);
rfsIm = rfs(end);


mkdir(fullfile('feats', dbname))
db_feats = all_db_feats(:,:,:, 1);
db_labels = all_db_labels(:,:,:, 1);
save(fullfile('feats', dbname, 'israeli_001.mat'), ...
  'db_feats', 'db_labels', 'feat_dims', ...
  'rfsIm', 'trace_H', 'trace_W', '-v7.3')
for i=2:size(all_db_feats, 4)
  db_feats = all_db_feats(:,:,:, i);
  db_labels = all_db_labels(:,:,:, i);
  save(fullfile('feats', dbname, sprintf('israeli_%03d.mat', i)), ...
    'db_feats', 'db_labels', '-v7.3');
end
end
