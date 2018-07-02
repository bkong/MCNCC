function gen_resnetfeats_aerial(db_ind)
imscale = 0.25;

if nargin<1
  db_ind = 2;
end


[db_attr, db_chunks, dbname] = get_db_attrs('maps', db_ind);

trace_H = 128;
trace_W = 128;
facade_ims = zeros(trace_H, trace_W, 3, 606*3, 'single');
label_ims = zeros(trace_H, trace_W, 1, 606*3, 'single');
files = dir('datasets/facades/*.png');
for f=1:numel(files)
  [~, name, ~] = fileparts(files(f).name);
  facade = imread(fullfile('facades', sprintf('%s.jpg', name)));
  label = imread(fullfile('facades', sprintf('%s.png', name)));

  [height, width, ~] = size(facade);
  if height<width
    facade = imresize(facade, [128 384]);
    label  = imresize(label, [128 384], 'nearest');

    % left
    facade_ims(:,:,:, (f-1)*3+1) = facade(:, 1:128, :);
    label_ims(:,:,:, (f-1)*3+1)  = label(:, 1:128);
    % center
    facade_ims(:,:,:, (f-1)*3+2) = facade(:, 129:256, :);
    label_ims(:,:,:, (f-1)*3+2)  = label(:, 129:256);
    % right
    facade_ims(:,:,:, (f-1)*3+3) = facade(:, 257:end, :);
    label_ims(:,:,:, (f-1)*3+3)  = label(:, 257:end);
  else
    facade = imresize(facade, [384 128]);
    label  = imresize(label, [384 128], 'nearest');

    % top
    facade_ims(:,:,:, (f-1)*3+1) = facade(1:128, :,:);
    label_ims(:,:,:, (f-1)*3+1)  = label(1:128, :);
    % middle
    facade_ims(:,:,:, (f-1)*3+2) = facade(129:256, :,:);
    label_ims(:,:,:, (f-1)*3+2)  = label(129:256, :);
    % bottom
    facade_ims(:,:,:, (f-1)*3+3) = facade(257:end, :,:);
    label_ims(:,:,:, (f-1)*3+3)  = label(257:end, :);
  end
end

% throw out patches that are more than 50% background
inds = find(sum(sum(label_ims==1, 1),2)>(128*128*0.5));
facade_ims(:,:,:, inds) = [];
label_ims(:,:,:, inds) = [];

% throw out patches where the labels are nearly identical
% (i.e., more than 99% correlated), but keep one copy
test = single(reshape(label_ims, 128*128, [])');
test = bsxfun(@minus, test, mean(test));
test = bsxfun(@rdivide, test, sqrt(sum(test.*test, 2)));
test_corr = test*test' + diag(NaN(size(label_ims,4), 1));
inds = find(test_corr(:)>=0.99);
[rows,cols] = ind2sub(size(test_corr), inds);
inds = [];
for i=unique(cols)'
  if any(inds==i), continue, end

  match_inds = find(cols==i);
  inds = cat(1, inds, rows(match_inds));
end
facade_ims(:,:,:, inds) = [];
label_ims(:,:,:, inds) = [];

% expand the entire dynamic range
color_labels = zeros(size(label_ims, 1), size(label_ims, 2), 3, size(label_ims, 4), 'uint8');
for i=1:size(label_ims,4)
  label = label_ims(:,:,:, i);
  color_labels(:,:,:, i) = im2uint8(reshape(cmap(label(:), :), [size(label, 1), size(label, 2), 3]));
end
label_ims = color_labels;
clear color_labels

% zero-center the data
mean_im = mean(facade_ims, 4);
mean_im_pix = mean(mean(mean_im,1), 2);
facade_ims = bsxfun(@minus, facade_ims, mean_im_pix);
mean_im = mean(label_ims, 4);
mean_im_pix = mean(mean(mean_im,1), 2);
label_ims = bsxfun(@minus, label_ims, mean_im_pix);


% load and modify network
flatnn = load('imagenet-resnet-50-dag.mat');
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
  % generate features for facade images
  data = {'data', facade_ims(:,:,:, db_chunks{c})};
  all_db_feats = generate_db_CNNfeats(net, data);

  for i=1:size(all_db_feats, 4)
    db_feats = all_db_feats(:,:,:, i);
    save(fullfile('feats', dbname, sprintf('facade_%04d.mat', db_chunks{c}(i))), ...
         'db_feats', '-v7.3');
  end
  clear all_db_feats

  % generate features for map images
  data = {'data', label_ims(:,:,:, db_chunks{c})};
  all_db_feats = generate_db_CNNfeats(net, data);

  for i=1:size(all_db_feats, 4)
    db_feats = all_db_feats(:,:,:, i);
    save(fullfile('feats', dbname, sprintf('label_%04d.mat', db_chunks{c}(i))), ...
         'db_feats', '-v7.3');
  end
  clear all_db_feats
end
end
