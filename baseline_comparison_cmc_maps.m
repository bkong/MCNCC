function dist_comparison_cmc_maps(db_ind, is_train)
if nargin<1
  db_ind = 2;
end
if nargin<2
  is_train = true;
end

dists = {'cosine', 'euclidean', '3dncc', 'mcncc'};
[~, ~, dbname] = get_db_attrs('maps', db_ind);


blacklist_cols = [25 84 100 107 151 187 336 443 465 485 563 590 593 601 619 ...
                  642 644 668 672 783 923 955 969 995 1021 1038 1044 1055 ...
                  1075 1683 2054];
if is_train
  blacklist_rows = blacklist_cols(blacklist_cols<=1097);
  labels = 1:(1097-numel(blacklist_rows));
else
  blacklist_rows = blacklist(blacklist_cols>1097);
  labels = (1097-numel(blacklist_rows)+1):(2194-numel(blacklist_cols));
end

cmcs = zeros(numel(dists), 2194-numel(blacklist_cols));
for d=1:numel(dists)
  load(fullfile('results', dbname, sprintf('maps_%s_similarity.mat', dists{d})))
  if is_train
    similarity = similarity(1:1097, :);
    % blacklist
    similarity(blacklist_rows, :) = [];
  else
    similarity = similarity(1098:end, :);
    % blacklist
    similarity(blacklist_rows-1097, :) = [];
  end
  similarity(:, blacklist_cols) = [];

  % compute cmc for each row
  for q=1:size(similarity, 1)
    [~, inds] = sort(similarity(q, :), 'descend');
    cmcs(d, :) = cmcs(d, :)+( cumsum(inds==labels(q))>0 );
  end
end
cmcs = cmcs./(1097-numel(blacklist_rows)).*100;


set(0,'DefaultAxesFontName', 'Times New Roman')
set(0,'DefaultTextFontname', 'Times New Roman')
font_size = 48;
line_width = 6;

colors = get(gca, 'colororder');
hold on
for d=1:numel(dists)
  plot([1:2194-numel(blacklist_cols)]./(2194-numel(blacklist_cols)).*100, cmcs(d, :), ...
       'LineWidth', line_width, ...
       'Color', colors(d, :))
end
hold off
grid on
xlim([0 5]), ylim([0 100])
xlabel('# database images reviewed (%)')
ylabel('# correct matches (%)')
lgd = legend('Cosine', 'Euclidean', ...
             '$[\mu,\sigma]$', '$[\mu_c,\sigma_c]$', ...
             'Location', 'NorthEastOutside');
lgd.Interpreter = 'latex';
axis square
set(gca, 'FontSize', font_size)
set(findall(gcf,'type','text'), 'FontSize', font_size)

end
