function dist_comparison_cmc_facades(db_ind, is_train)
if nargin<1
  db_ind = 2;
end
if nargin<2
  is_train = true;
end

dists = {'cosine', 'euclidean', '3dncc', 'mcncc'};
[~, ~, dbname] = get_db_attrs('facades', db_ind);


if is_train
  labels = 1:825;
else
  labels = 826:1657;
end

cmcs = zeros(numel(dists), 1657);
for d=1:numel(dists)
  load(fullfile('results', dbname, sprintf('facades_%s_similarity.mat', dists{d})))
  if is_train
    similarity = similarity(1:825, :);
  else
    similarity = similarity(826:end, :);
  end

  % compute cmc for each row
  for q=1:size(similarity, 1)
    [~, inds] = sort(similarity(q, :), 'descend');
    cmcs(d, :) = cmcs(d, :)+( cumsum(inds==labels(q))>0 );
  end
end
cmcs = cmcs./numel(labels).*100;


set(0,'DefaultAxesFontName', 'Times New Roman')
set(0,'DefaultTextFontname', 'Times New Roman')
font_size = 48;
line_width = 6;

colors = get(gca, 'colororder');
hold on
for d=1:numel(dists)
  plot([1:1657]./1657.*100, cmcs(d, :), ...
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
