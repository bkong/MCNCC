function [db_attr,db_chunks,dbname] = get_db_attrs(dataset, db_ind, info)

if nargin<4 || isempty(info)
  info_inds = 1:6;
else
  all_info = {'suffix', 'layer', 'pool_size', 'pool_stride', 'pool_pad', 'im_pad'};
  info_inds = find(ismember(all_info, info));
end

% db_attr = {suffix, layer to cut network, pool size, pool stride, pool pad, img pad}
switch db_ind
  case 0
    db_attr = {'raw', 'conv1', 49, 1, 24, 24};
  case 1
    db_attr = {'2x',  'pool1', 21, 1, 10, 23};
  case 2
    db_attr = {'4x', 'res2c_branch2a', 5, 1, 2, 21};
  case 3
    db_attr = {'8x', 'res3d_branch2a', 1, 1, 0, 41};
  case 4
    db_attr = {'16x', 'res4d_branch2a', 1, 1, 0, 97};
end
dbname = sprintf('resnet_db_%s', db_attr{1}) ;

if strcmpi(dataset,'israeli')
  db_chunks = {1:387};
elseif strcmpi(dataset,'fid300')
  db_chunks = {1:1175};
elseif strcmpi(dataset,'facades')
  db_chunks = {1:1657};
elseif strcmpi(dataset,'maps')
  db_chunks = {1:1097, 1098:2194};
else
  error(sprintf('Dataset: %s is not valid!\n', dataset))
end

db_attr = db_attr(info_inds);

end
