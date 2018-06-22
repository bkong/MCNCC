function db = generate_db_CNNfeats(net, data)
% assume the network's last variable (x) is the features
% if x is HxWxKxN, then there are H*W number of patches for each data point

net.mode = 'test';
% run the network
net.move('gpu');
% convert from RGB to grayscale
ims = gather(data{2});
gray = zeros(size(ims,1), size(ims,2), 3, size(ims,4), 'like', ims);
for i=1:size(ims,4)
  gray(:,:,:,i) = repmat(mean(ims(:,:,:,i),3), 1,1,3);
end

data{2} = gpuArray(zeros(size(gray,1), size(gray,2), size(gray,3), 'single'));
net.eval(data);
resp = net.vars(end).value;
feats = zeros(size(resp,1), size(resp,2), size(resp,3), size(gray,4), 'single');

siz = 10;
num = ceil(size(gray,4)/siz);
for b=1:num
  lef = (b-1)*siz+1;
  rig = min((b-1)*siz+siz, size(gray,4));
  data{2} = gpuArray(gray(:,:,:,lef:rig));
  net.eval(data);

  feats(:,:,:,lef:rig) = gather(net.vars(end).value);
end

db = feats ;

end
