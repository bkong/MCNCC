% JSS3 - 2016-10-02
% find an affine transform from the feature map space
% to the image space
function [t] = feat_2_image(rfs) % feat2hm feat2hm
    if isstruct(rfs)
        rfs = {rfs};
    end

    feat_coords_x1 = [-100 -100 +100 +100];
    feat_coords_y1 = [-100 +100 -100 +100];    
    feat_coords_x2 = [-100 -100 +100 +100];
    feat_coords_y2 = [-100 +100 -100 +100];
    for rfIter = 1:numel(rfs)
        rf = rfs{rfIter};
        if numel(rf) > 1
            rf = rf(end);
        end
        feat_coords_x2  = rf.stride(2).*(feat_coords_x2-1) + rf.offset(2);
        feat_coords_y2  = rf.stride(1).*(feat_coords_y2-1) + rf.offset(1);
    end
    
    t = fitgeotrans([feat_coords_x1; feat_coords_y1]',[feat_coords_x2; feat_coords_y2]','affine');
end
