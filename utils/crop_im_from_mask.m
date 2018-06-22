function crop = crop_im_from_mask(im, mask)
  bbox = mask2bbox(mask);
  crop = im(bbox(2):bbox(4), bbox(1):bbox(3), :);
end


function bbox = mask2bbox(mask)
% bbox = [x1 y1 x2 y2]
  [rows,cols] = find(mask);
  bbox = [min(cols) min(rows) max(cols) max(rows)];
end
