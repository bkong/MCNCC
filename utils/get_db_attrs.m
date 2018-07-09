function [db_attr,db_chunks,dbname] = get_db_attrs(dataset, db_ind, info)

if nargin<4 || isempty(info)
  info_inds = 1:2;
else
  all_info = {'suffix', 'layer'};
  info_inds = find(ismember(all_info, info));
end

% db_attr = {suffix, layer to cut network, pool size, pool stride, pool pad, img pad}
switch db_ind
  case 0
    db_attr = {'pixel_raw', 'conv1'};
  case 1
    db_attr = {'resnet_db_2x',  'pool1'};
  case 2
    db_attr = {'resnet_db_4x', 'res2c_branch2a'};
  case 3
    db_attr = {'resnet_db_8x', 'res3d_branch2a'};
  case 4
    db_attr = {'resnet_db_16x', 'res4d_branch2a'};
end
dbname = db_attr{1};

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
