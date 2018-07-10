function preprocess_israeli
imscale = 0.25;

max_H = 2408;
max_W = 1497;
subdirs = dir(fullfile('datasets', 'israeli'));
subdirs(1:2) = [];
subdirs(find(~[subdirs(:).isdir])) = [];
numTraces = numel(subdirs);

treadnames = {};
numPrints = 0;
for d=1:numTraces
  anno_file = fullfile(subdirs(d).folder, subdirs(d).name, ...
                       'trace_annotations.mat');
  if exist(anno_file, 'file')
    load(anno_file)
    treadnames{end+1} = treadid;
  else
    treadnames{end+1} = subdirs(d).name;
  end

  files = dir(fullfile(subdirs(d).folder, subdirs(d).name, ...
                       '*print*_alignment.mat'));
  numPrints = numPrints+numel(files);
end
% convert treadnames to treadids
uniqnames = reshape(unique(treadnames), 1, []);
% get a count of the tread groups so we can sort them
treadhist = zeros(numel(uniqnames), 1);
for u=1:numel(uniqnames)
  inds = find(ismember(treadnames, uniqnames{u}));
  treadhist(u) = numel(inds);
end
[treadhist, inds] = sort(treadhist, 'descend');
uniqnames = uniqnames(inds);
% tread group id by group size id=1 is the largest group
trace_treadids = zeros(size(treadnames));
for u=1:numel(uniqnames)
  inds = find(ismember(treadnames, uniqnames{u}));
  trace_treadids(inds) = u;
end

trace_H = floor(max_H*0.5);
trace_W = floor(max_W*0.5);
all_trace_ims = zeros(trace_H, trace_W, 3, numTraces, 'uint8');
all_trace_dnames = cell(1, numTraces);
all_print_ims = zeros(trace_H, trace_W, 3, numPrints, 'uint8');
all_print_masks = zeros(trace_H, trace_W, 1, numPrints, 'logical');
all_print_treadids = zeros(1, numPrints);
all_print_dnames = cell(1, numPrints);
cnt = 0;
for d=1:numTraces
  trace_im = imresize(imread(fullfile(subdirs(d).folder, subdirs(d).name, ...
                                      'trace_image.jpg')), ...
                      0.3);
  % flip left shoes to mimic right shoes
  is_left = subdirs(d).name(end)=='L';
  if is_left
    trace_im = fliplr(trace_im);
  end
  Rfixed = imref2d(size(trace_im)); % registration on original unpadded image
  trace_im = pad_im(trace_im, max_H, max_W);

  %
  % Collect crime scene prints and do initial registration based on cpselect points
  %
  files = dir(fullfile(subdirs(d).folder, subdirs(d).name, ...
                       '*print*_alignment.mat'));
  print_ims = cell(numel(files), 1);
  print_masks = cell(numel(files), 1);
  for f=1:numel(files)
    load(fullfile(files(f).folder, files(f).name), ...
         'print_pts', 'trace_pts', 'flip_contrast')
    % print and trace images were registered at imresize(..., 0.3)
    print_im = imresize(imread(fullfile(files(f).folder, ...
                                        sprintf('%s.jpg', files(f).name(1:end-14)))), ...
                        0.3);

    if size(print_im, 3)==1
      print_im = cat(3, print_im, print_im, print_im);
    end
    if flip_contrast
      print_im = 1-im2double(print_im);
      print_im = im2uint8(print_im);
    end
    print_mask = ones(size(print_im, 1), size(print_im, 2), 'logical');

    % register print_im to trace_im using cpselect points
    tform = fitgeotrans(print_pts, trace_pts, 'affine');
    print_registered = imwarp(print_im, tform, 'OutputView', Rfixed, 'FillValues', 255);
    print_mask_registered = imwarp(print_mask, tform, 'OutputView', Rfixed, 'FillValues', 0);

    % flip left shoes to mimic right shoes
    if is_left
      print_registered = fliplr(print_registered);
      print_mask_registered = fliplr(print_mask_registered);
    end

    print_ims{f} = pad_im(print_registered, max_H, max_W, 255);
    print_masks{f} = pad_im(print_mask_registered, max_H, max_W, 0);
  end

  [rot_trace, rot_prints, rot_masks] = rotate_trace(d, trace_im, print_ims, print_masks);
  [rot_trace, rot_prints, rot_masks] = crop_trace(rot_trace, rot_prints, rot_masks);

  all_trace_ims(:,:,:, d) = resize_and_pad(rot_trace, trace_H, trace_W);
  all_trace_dnames{d} = subdirs(d).name;
  for p=1:numel(rot_prints)
    all_print_ims(:,:,:, cnt+p) = resize_and_pad(rot_prints{p}, trace_H, trace_W);
    all_print_masks(:,:, 1, cnt+p) = resize_and_pad(rot_masks{p}, trace_H, trace_W, 0);
    all_print_treadids(cnt+p) = trace_treadids(d);
    all_print_dnames{cnt+p} = subdirs(d).name;
  end
  cnt = cnt+numel(rot_prints);
end


%
% Align test impressions within the same class and apply the same transformation
% to corresponding crime scene prints and their masks.
%
uniqueids = unique(trace_treadids);
% convert from RGB to grayscale
trace_ims = zeros(trace_H, trace_W, 1, numTraces, 'like', all_trace_ims);
for i=1:numTraces
  trace_ims(:,:, 1, i) = rgb2gray(all_trace_ims(:,:,:, i));
end

% configure optimizer
[opt, metric] = imregconfig('monomodal');
optRig = opt;
optSim = opt;
optSim.MaximumStepLength = 0.0625/2;
optAff = opt;
optAff.MaximumStepLength = 0.0625/2;
imref = imref2d([trace_H trace_W]);

for id=uniqueids
  id_inds = find(trace_treadids==id);

%  fprintf('id=%d: ', id)
  % use the first image as the base
  fixed_im = trace_ims(:,:,:, id_inds(1));
  for t=id_inds
%    fprintf('%d', t)

    [moving_pts, fixed_pts] = manual_align(t);
    if isempty(moving_pts)
      % learn transformation on images with zero mean
      moving_im = trace_ims(:,:,:, t);

      tform = affine2d;
      prev_err = 1e10;
      optRig.MaximumIterations = 100;
      done = false;
      while ~done
        tform = imregtform(moving_im, fixed_im, 'rigid', optRig, metric, ...
                           'PyramidLevels', 1, ...
                           'InitialTransformation', tform);
%        fprintf('r')

        res_im = imwarp(moving_im, tform, ...
                        'OutputView', imref, 'FillValues', 255);

        err = sqrt(sum( (fixed_im(:)-res_im(:)).^2 ));
        if err==0 || (prev_err-err)/max(prev_err, err) <= 1e-3
          done = true;
          optRig.MaximumIterations = 20;
        end
        if ~tform.isRigid
          tform = prev_tform;
          done = true;
        end

        prev_err = err;
        prev_tform = tform;
      end

      optSim.MaximumIterations = 100;
      done = false;
      while ~done
        tform = imregtform(moving_im, fixed_im, 'similarity', optSim, metric, ...
                           'PyramidLevels', 1, ...
                           'InitialTransformation', tform);
%        fprintf('s')

        res_im = imwarp(moving_im, tform, ...
                        'OutputView', imref, 'FillValues', 255);

        err = sqrt(sum( (fixed_im(:)-res_im(:)).^2 ));
        if err==0 || (prev_err-err)/max(prev_err, err) <= 1e-3
          done = true;
        else
          optSim.MaximumIterations = 20;
        end
        if ~tform.isSimilarity
          tform = prev_tform;
          done = true;
        end

        prev_err = err;
        prev_tform = tform;
      end

      optAff.MaximumIterations = 100;
      done = false;
      while ~done
        tform = imregtform(moving_im, fixed_im, 'affine', optAff, metric, ...
                           'PyramidLevels', 1, ...
                           'InitialTransformation', tform);
%        fprintf('a')

        res_im = imwarp(moving_im, tform, ...
                        'OutputView', imref, 'FillValues', 255);

        err = sqrt(sum( (fixed_im(:)-res_im(:)).^2 ));
        if err==0 || (prev_err-err)/max(prev_err, err) <= 1e-3
          done = true;
        else
          optAff.MaximumIterations = 20;
        end

        prev_err = err;
      end
%      fprintf(' ')

    else
      tform = fitgeotrans(moving_pts, fixed_pts, 'affine');
%      fprintf('m ')
    end

    % apply tranformation on the original test impression
    all_trace_ims(:,:,:, t) = imwarp(all_trace_ims(:,:,:, t), tform, ...
                                     'OutputView', imref, 'FillValues', 255);
    % same for crime scene print(s) if they exist
    print_ids = find(ismember(all_print_dnames, all_trace_dnames{t}));
    for p=print_ids
      all_print_ims(:,:,:, p) = imwarp(all_print_ims(:,:,:, p), tform, ...
                                       'OutputView', imref, 'FillValues', 255);
      all_print_masks(:,:,:, p) = imwarp(all_print_masks(:,:,:, p), tform, ...
                                         'OutputView', imref, 'FillValues', 0);
    end
%    fprintf('\n')
  end
end
data = struct('trace_ims', imresize(all_trace_ims, imscale), ...
              'trace_treadids', trace_treadids, ...
              'print_ims', imresize(all_print_ims, imscale), ...
              'print_masks', imresize(all_print_masks, imscale, 'nearest'), ...
              'print_treadids', all_print_treadids);
mkdir(fullfile('datasets', 'israeli'))
save(fullfile('datasets', 'israeli', 'preprocessed_data.mat'), '-struct', 'data')
end

% Same function as resize_and_pad(), but behaves slightly differently.
% Keep using so subsequent code gives the original result.
function im = pad_im(im, newH, newW, padval)
  if nargin<4, padval = 0; end
  if abs(size(im, 1)-newH)<abs(size(im, 2)-newW)
    im = imresize(im, [newH NaN]);
  else
    im = imresize(im, [NaN newW]);
  end

  difH = newH-size(im, 1);
  difW = newW-size(im, 2);
  % imwarp fills with zeros, so use padarray default padval=0
  im = padarray(im, [floor(difH/2) floor(difW/2)], padval, 'pre');
  im = padarray(im, [ceil(difH/2) ceil(difW/2)], padval, 'post');
end

function im = manual_fix(im, idx)
  if idx==15
    im(1:100    , 1:400   , :)  = 255;
  elseif idx==16
    im(700:end  , 1160:end, :)  = 255;
    im(1070:end , 1120:end, :)  = 255;
  elseif idx==20
    im(1:1045   , 1:422   , :)  = 255;
    im(1:1120   , 1210:end, :)  = 255;
    im(1:530    , 1174:end, :)  = 255;
    im(1030:1050, 1184:end, :)  = 255;
  elseif idx==21
    im(:        , 1:350   , :)  = 255;
    im(1135:end , 1160:end, :)  = 255;
  elseif idx==23
    im(700:780  , 1:300   , :)  = 255;
  elseif idx==28
    im(1350:1550, 1110:end, :)  = 255;
    im(2230:end , 1110:end, :)  = 255;
  elseif idx==43
    im(:        , 1:330   , :)  = 255;
    im(1:140    , 1:440   , :)  = 255;
  elseif idx==44
    im(700:800  , 1160:end, :)  = 255;
  elseif idx==47
    im(1:230    , 1:450   , :)  = 255;
    im(1:500    , 1:350   , :)  = 255;
  elseif idx==59
    im(1:230    , 1:450   , :)  = 255;
    im(1:500    , 1:350   , :)  = 255;
  elseif idx==61
    im(1:230    , 1:400   , :)  = 255;
  elseif idx==62
    im(1:150    , 1:600   , :)  = 255;
    im(1:230    , 1:400   , :)  = 255;
  elseif idx==103
    im(1200:end , 200:400 , :) = 255;
  elseif idx==112
    im(1:230    , 1:400   , :) = 255;
    im(2360:end , :       , :) = 255;
  elseif idx==122
    im(1:320    , 1:340   , :) = 255;
    im(:        , 1200:end, :) = 255;
  elseif idx==127
    im(1:500    , 1:350   , :) = 255;
    im(1:300    , 1150:end, :) = 255;
  elseif idx==132
    im(1:260    , 1:500   , :) = 255;
  elseif idx==144
    im(1:164    , :       , :) = 255;
    im(1:380    , 1:460   , :) = 255;
  elseif idx==153
    im(1:310    , 1:450   , :) = 255;
  elseif idx==159
    im(1:350    , 1:450   , :) = 255;
    im(1:50     , :       , :) = 255;
  elseif idx==160
    im(1:510    , 1:390   , :) = 255;
  elseif idx==163
    im(1:150    , 1:410   , :) = 255;
  elseif idx==164
    im(1:200    , 1:490   , :) = 255;
    im(1:410    , 1:405   , :) = 255;
  elseif idx==169
    im(1:50     , :       , :) = 255;
    im(2376:end , :       , :) = 255;
  elseif idx==175
    im(1:110    , 1035:end, :) = 255;
  elseif idx==178
    im(1:200    , 1:580   , :) = 255;
  elseif idx==187
    im(:        , 1:300   , :) = 255;
    im(:        , 1210:end, :) = 255;
    im(1:100    , :       , :) = 255;
    im(2390:end , :       , :) = 255;
    im(2220:end , 1:470   , :) = 255;
    im(1:245    , 1:410   , :) = 255;
    im(1:310    , 1010:end, :) = 255;
  elseif idx==193
    im(1:1465   , 1198:end, :) = 255;
    im(1:1700   , 1258:end, :) = 255;
  elseif idx==195
    im(1:75     , :       , :) = 255;
  elseif idx==201
    im(1:170    , 1:400   , :) = 255;
  elseif idx==209
    im(2220:end , :       , :) = 255;
  elseif idx==214
    im(1:200    , 1:400   , :) = 255;
  elseif idx==224
    im(1:130    , 1:400   , :) = 255;
  elseif idx==226
    im(1:130    , 1:400   , :) = 255;
  elseif idx==232
    im(1:130    , 1:400   , :) = 255;
    im(2364:end , :       , :) = 255;
  elseif idx==233
    im(1:390    , 1010:end, :) = 255;
    im(2360:end , :       , :) = 255;
  elseif idx==236
    im(1:130    , 1:600   , :) = 255;
  elseif idx==239
    im(:        , 1:350   , :) = 255;
    im(1:300    , 1:450   , :) = 255;
  elseif idx==250
    im(1:344    , 1:460   , :) = 255;
  elseif idx==252
    im(1:100    , 1:400   , :) = 255;
  elseif idx==255
    im(1:125    , 1:400   , :) = 255;
    im(1400:1500, 1100:end, :) = 255;
  elseif idx==258
    im(1:170    , 1:480   , :) = 255;
  elseif idx==264
    im(:        , 1240:end, :) = 255;
  elseif idx==265
    im(1:200    , 1:400   , :) = 255;
  elseif idx==275
    im(1:50     , :       , :) = 255;
    im(1:70     , 1:500   , :) = 255;
    im(1:240    , 1:390   , :) = 255;
  elseif idx==276
    im(1:70     , 1:400   , :) = 255;
  elseif idx==278
    im(1:190    , 1:500   , :) = 255;
  elseif idx==279
    im(2168:end , 1018:end, :) = 255;
  elseif idx==284
    im(1:265    , 1:450   , :) = 255;
    im(2370:end , :       , :) = 255;
    im(2040:2150, 1000:end, :) = 255;
  elseif idx==287
    im(1:344    , 1:460   , :) = 255;
  elseif idx==305
    im(700:760  , 1230:end, :) = 255;
  elseif idx==311
    im(1:160    , 1:370   , :) = 255;
  elseif idx==316
    im(2320:end , 1100:end, :) = 255;
  elseif idx==318
    im(1700:1800, 1020:end, :) = 255;
  elseif idx==324
    im(1:14     , :       , :) = 255;
    im(1:130    , 900:end , :) = 255;
  elseif idx==329
    im(2370:end , :       , :) = 255;
  elseif idx==333
    im(1:550    , 1:400   , :) = 255;
  elseif idx==349
    im(1:120    , 1200:end, :) = 255;
  elseif idx==354
    im(20:40    , 1118:end, :) = 255;
  elseif idx==361
    im(1840:1940, 250:310 , :) = 255;
  elseif idx==364
    im(1:70     , 900:1000, :) = 255;
  elseif idx==373
    im(1:686    , 1:340   , :) = 255;
  elseif idx==375
    im(1:304    , 1:438   , :) = 255;
    im(2296:end , 1016:end, :) = 255;
  elseif idx==379
    im(1:180    , 1:400   , :) = 255;
  elseif idx==384
    im(1:180    , 1:400   , :) = 255;
  end
end

function [moving_pts, fixed_pts] = manual_align(idx)
  moving_pts = [];
  fixed_pts = [];
  if idx==38
    % align idx=38 to idx=29
    fixed_pts = 1e3.*[...
    0.3880    0.1230; ...
    0.2140    0.5600; ...
    0.4440    0.9200; ...
    0.2480    1.1280];
    moving_pts = 1e3.*[...
    0.3830    0.0650; ...
    0.1970    0.5230; ...
    0.4580    0.8970; ...
    0.2460    1.1210];
  elseif idx==87
    % align idx=87 to idx=29
    fixed_pts = 1e3.*[...
    0.2480    1.1280; ...
    0.4440    0.9200; ...
    0.2110    0.4900; ...
    0.5610    0.4110];
    moving_pts = 1e3.*[...
    0.2590    1.1390; ...
    0.4500    0.9390; ...
    0.2120    0.5290; ...
    0.5610    0.4630];
  elseif idx==107
    % align idx=107 to idx=53
    fixed_pts = 1e3.*[...
    0.3758    1.0315; ...
    0.1982    0.4355; ...
    0.5455    0.5515; ...
    0.3837    0.0882];
    moving_pts = 1e3.*[...
    0.3792    1.0435; ...
    0.2042    0.4702; ...
    0.5402    0.5782; ...
    0.3812    0.1327];
  elseif idx==112
    % align idx=112 to idx=10
    fixed_pts = 1e3.*[...
    0.3780    1.0450; ...
    0.4940    0.8710; ...
    0.2385    0.2833; ...
    0.4833    0.1681];
    moving_pts = 1e3.*[...
    0.3690    1.0630; ...
    0.4830    0.8930; ...
    0.2433    0.3225; ...
    0.4833    0.2145];
  elseif idx==131
    % align idx=131 to idx=12
    fixed_pts = 1e3.*[...
    0.3137    1.0697; ...
    0.2273    0.5745; ...
    0.4993    0.4609; ...
    0.4233    0.1345];
    moving_pts = 1e3.*[...
    0.3049    1.1665; ...
    0.2049    0.6217; ...
    0.5009    0.5057; ...
    0.4313    0.1513];
  elseif idx==207
    % align idx=207 to idx=33
    fixed_pts = [...
    292.0000  889.0000; ...
    401.0000  614.0000; ...
    452.0000  387.0000; ...
    295.0000  190.0000];
    moving_pts = [...
    288.0000  851.0000; ...
    394.0000  578.0000; ...
    443.0000  352.0000; ...
    289.0000  148.0000];
  elseif idx==244
    % align idx=244 to idx=18
    fixed_pts = [...
    296.0000  115.0000; ...
    471.0000  323.0000; ...
    336.0000  876.0000; ...
    470.0000  996.0000];
    moving_pts = 1e3.*[...
    0.2980    0.1510; ...
    0.4670    0.3570; ...
    0.3240    0.8960; ...
    0.4560    1.0140];
  elseif idx==245
    % align idx=245 to idx=25
    fixed_pts = 1e3.*[...
    0.2440    1.1340; ...
    0.4450    0.9220; ...
    0.3580    0.0230; ...
    0.1700    0.4810];
    moving_pts = 1e3.*[...
    0.2630    1.1330; ...
    0.4480    0.9280; ...
    0.3530    0.0850; ...
    0.2010    0.5180];
  elseif idx==276
    % align idx=276 to idx=23
    fixed_pts = 1e3.*[...
    0.2833    1.1643; ...
    0.5403    0.8608; ...
    0.2878    0.2953; ...
    0.5383    0.5058];
    moving_pts = 1e3.*[...
    0.3033    1.1698; ...
    0.5533    0.8553; ...
    0.2893    0.2863; ...
    0.5443    0.4933];
  elseif idx==281
    % align idx=281 to idx=280
    fixed_pts = 1e3.*[...
    0.2601    0.2681; ...
    0.5201    0.3745; ...
    0.3425    0.8137; ...
    0.3449    1.1473];
    moving_pts = 1e3.*[...
    0.2641    0.2657; ...
    0.5297    0.3577; ...
    0.3705    0.8025; ...
    0.3777    1.1313];
  elseif idx==282
    % align idx=282 to idx=176
    fixed_pts = 1e3.*[...
    0.5058    0.2388; ...
    0.2215    0.8682; ...
    0.4555    1.0742; ...
    0.3428    0.1988];
    moving_pts = 1e3.*[...
    0.5052    0.2295; ...
    0.2235    0.8368; ...
    0.4508    1.0395; ...
    0.3488    0.1902];
  elseif idx==286
    % align idx=286 to idx=285
    fixed_pts = 1e3.*[...
    0.2792    0.1662; ...
    0.1742    0.4255; ...
    0.2748    1.0115; ...
    0.4258    1.0648];
    moving_pts = 1e3.*[...
    0.2965    0.1968; ...
    0.1862    0.4548; ...
    0.2822    1.0548; ...
    0.4332    1.1058];
  elseif idx==361
    % align idx=361 to idx=360
    fixed_pts = [...
    341.2500   86.7500; ...
    301.2500  771.7500; ...
    527.2500  572.7500; ...
    482.2500  917.7500];
    moving_pts = [...
    337.7500  118.7500; ...
    298.7500  790.7500; ...
    520.2500  596.7500; ...
    474.7500  928.7500];
  elseif idx==374
    % align idx=374 to idx=221
    fixed_pts = 1e3.*[...
    0.3402    0.1777; ...
    0.4503    0.3342; ...
    0.2782    0.6102; ...
    0.4243    1.0248];
    moving_pts = [...
    347.2500  138.2500; ...
    451.2500  289.7500; ...
    283.2500  547.2500; ...
    422.7500  947.7500];
  end
end

function thresh = trace_thresholds(idx)
  switch idx
    case {3 5 6 10 34 39 49 50 55 57 58 65 66 72 86 87 91 106 113 149 154 155 159 166 177 179 181 188 189 197 198 199 210 215 216 217 218 219 223 224 227 236 237 243 244 249 251 254 255 260 265 273 277 284 286 290 292 293 301 311 313 319 324 330 335 336 344 345 351 352 365 370 371 379 385}
      thresh = [65 0 0];
    case {12 31 61 63 160 229 234 241 252 267 321 350 364 382}
      thresh = [70 0 0];
    case {14 15 17 28 54 117 139 140 170 209 214 221 257 258 275 278 312 384}
      thresh = [75 0 0];
    case {4 16 20 30 46 135 158 169 195 200 201 208 240 259 264 288 360 361 376}
      thresh = [80 0 0];
    case {43 47 59 67 112 127 132 164 226 232 296 323}
      thresh = [85 0 0];
    case {23 44 62 122 153 163 175 178 233 239 250 276 305 316 318 333 349 354}
      thresh = [86 0 0];
    case 20
      thresh = [70 30 30];
    case 187
      thresh = [88 30 51];
    case 266
      thresh = [65 28 40];
    case 280
      thresh = [75 35 0];
    case 287
      thresh = [65 0 50];
    case 329
      thresh = [68 27 30];
    case 375
      thresh = [71 44 0];
    otherwise
      thresh = [65 30 30];
  end
end

function [conv_rough_seg, shoe_pixels] = segment_trace(trace_im, thresh)
  if nargin>1
    orange_thresh = thresh(1);
    red_thresh = thresh(2);
    magenta_thresh = thresh(3);
  end

  orange = [237 86 2]./255;
  orange = rgb2lab(orange);
  orange = reshape(orange, [1 1 3]);
  red = [.9412 .251 .09804];
  red = rgb2lab(red);
  red = reshape(red, [1 1 3]);
  magenta = [.9882 .502 .6471];
  magenta = rgb2lab(magenta);
  magenta = reshape(magenta, [1 1 3]);

  trace_im = double(trace_im)./255;
  [traceH, traceW, traceC] = size(trace_im);
  lab_trace = rgb2lab(trace_im);
  orange_dist = lab_dist(lab_trace, orange(:,:, 2), orange(:,:, 3));
  red_dist    = lab_dist(lab_trace, red(:,:, 2), red(:,:, 3));
  magenta_dist    = lab_dist(lab_trace, magenta(:,:, 2), magenta(:,:, 3));
  shoe_pixels = ...
      orange_dist < orange_thresh & ...
      red_dist > red_thresh & ...
      magenta_dist > magenta_thresh;
  % do some structring
  se = strel('disk', 3);
  shoe_pixels = imopen(shoe_pixels, se);
  shoe_pixels = conv2(double(shoe_pixels), ones(5, 5), 'same')./25 > .5;
  conv_rough_seg = bwconvhull(shoe_pixels);
end

function c_dist = lab_dist(lab_trace, a, b)
  [traceH, traceW, traceC] = size(lab_trace);
  a_diff = lab_trace(:,:, 2) - repmat(a, [traceH traceW 1]);
  b_diff = lab_trace(:,:, 3) - repmat(b, [traceH traceW 1]);
  c_dist = sqrt(a_diff.^2 + b_diff.^2);
end

function B = imrotate255(A, deg, fillValues)
  tform = affine2d([cosd(deg) -sind(deg) 0; sind(deg) cosd(deg) 0; 0 0 1]);
  RA = imref2d(size(A));
  Rout = images.spatialref.internal.applyGeometricTransformToSpatialRef(RA, tform);

  % Trim Rout, preserve center and resolution.
  Rout.ImageSize = RA.ImageSize;
  xTrans = mean(Rout.XWorldLimits) - mean(RA.XWorldLimits);
  yTrans = mean(Rout.YWorldLimits) - mean(RA.YWorldLimits);
  Rout.XWorldLimits = RA.XWorldLimits+xTrans;
  Rout.YWorldLimits = RA.YWorldLimits+yTrans;

  if nargin<3
    fillValues = 255;
  end
  B = imwarp(A, tform, 'bilinear', ...
             'OutputView', Rout, ...
             'SmoothEdges', true, ...
             'FillValues', fillValues);
end

function [rot_trace, rad] = align_trace(trace_im, rough_seg)
  [Y, X, V] = find(rough_seg);
  Y = Y - mean(Y);
  X = X - mean(X);
  coeff = pca([X Y]);
  if numel(coeff)>4
    coeff = pca([X Y]');
  end
  assert(numel(coeff)<=4);
  comp_1 = coeff(:, 1);
  rad = atan2(comp_1(1), comp_1(2));
  rot_trace = imrotate255(trace_im, -rad2deg(rad));
end

% Orient test impression so its major axis is vertical, and transform
% crime scene prints in the same way.
function [rot_trace, print_ims, print_masks] = ...
    rotate_trace(trace_idx, trace_im, print_ims, print_masks)
  trace_thresh = trace_thresholds(trace_idx);
  trace_im_seg = segment_trace(trace_im, trace_thresh);
  % do PCA to find the major axis.
  [rot_trace, trace_rad] = align_trace(trace_im, trace_im_seg);
  rot_trace_seg = segment_trace(rot_trace, trace_thresh);
  se = strel('disk', 5);
  rot_trace_seg = imdilate(rot_trace_seg, se);
  rot_trace(~repmat(rot_trace_seg, [1 1 3])) = 255;
  % rotate print images and masks the same way
  for p=1:numel(print_ims)
    print_ims{p} = imrotate255(print_ims{p}, -rad2deg(trace_rad));
    print_masks{p} = imrotate(print_masks{p}, -rad2deg(trace_rad), 'nearest', 'crop');
  end
end

% Crop test impressions minimize background pixels, and crop
% crime scene prints in the same way.
function [trace_im, print_ims, print_masks] = crop_trace(trace_im, print_ims, print_masks)
  grey_trace = rgb2gray(trace_im);
  y1 = min(find(any(grey_trace ~= 255 & grey_trace ~= 0, 2)));
  y2 = max(find(any(grey_trace ~= 255 & grey_trace ~= 0, 2)));
  x1 = min(find(any(grey_trace ~= 255 & grey_trace ~= 0, 1)));
  x2 = max(find(any(grey_trace ~= 255 & grey_trace ~= 0, 1)));
  trace_im = trace_im(y1:y2, x1:x2, :);
  for p = 1:numel(print_ims)
    print_ims{p} = print_ims{p}(y1:y2, x1:x2, :);
  end
  for p = 1:numel(print_masks)
    print_masks{p} = print_masks{p}(y1:y2, x1:x2, :);
  end
end

function im_final = resize_and_pad(im, H, W, fillValue)
  [H1, W1, C1] = size(im);
  sx = W./W1; sy = H./H1;
  if sx>sy
    im = imresize(im, sy);
  else
    im = imresize(im, sx);
  end
  if nargin<4
    fillValue = 255;
  end
  im_final = zeros(H, W, C1, 'like', im)+fillValue;
  cy = round(H./2); cx = round(W./2);
  [H1, W1, C1] = size(im);
  half_width = floor(W1./2);
  half_height = floor(H1./2);
  y_range = cy-half_height+1:cy-half_height+H1;
  x_range = cx-half_width+1:cx-half_width+W1;
%  im_final = zeros(max(H, cy-half_height+H1), ...
%                   max(W, cx-half_width+W1), ...
%                   C1, 'like', im)+fillValue;
  if numel(size(im))==2
    im_final(y_range, x_range) = im;
  else
    im_final(y_range, x_range, :) = im;
  end
  im_final = imresize(im_final, [H, W]);
end
