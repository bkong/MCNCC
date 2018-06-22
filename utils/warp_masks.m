function feat_masks = warp_masks(print_masks, im_f2i, feat_dims, db_ind)
  if db_ind==0
    feat_masks = print_masks;
  else
    feat_masks = imwarp(print_masks, im_f2i.invert(), 'nearest');
    if size(feat_masks,1)~=feat_dims(1) || size(feat_masks,2)~=feat_dims(2)
      if size(feat_masks,1)>feat_dims(1)
        top = 2; bot = feat_dims(1)+1;
      else
        top = 1; bot = feat_dims(1);
      end
      if size(feat_masks,2)>feat_dims(2)
        lef = 2; rig = feat_dims(2)+1;
      else
        lef = 1; rig = feat_dims(2);
      end
      feat_masks = feat_masks(top:bot, lef:rig, :,:);
    end
  end
end
