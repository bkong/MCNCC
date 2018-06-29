function Basel_compare_cmc_latent_masked(db_ind)
if nargin<1
  db_ind = 2;
end


[db_attr, db_chunks, dbname] = get_db_attrs('fid300', db_ind, {'suffix'});
db_chunk_inds = db_chunks{1};
db_labels = zeros(numel(db_chunk_inds), 1);
for i=1:numel(db_chunk_inds)
  dat = load(fullfile('feats', dbname, sprintf('fid300_%03d.mat', db_chunk_inds(i))));
  db_labels(i) = dat.db_labels;
end
load(fullfile('datasets', 'FID-300', 'label_table.mat'), 'label_table')


ncc_cmc = zeros(1, 1175);
for p=1:300
  % load patch results
  fname = fullfile('results', dbname, sprintf('alignment_search_ones_res_%04d.mat', p));
  load(fname, 'minsONES')

  query_label = label_table(p, 2);

  % NCC
  [~,inds] = sort(minsONES(:)', 'descend');
  assert(numel(inds)==1175);
  ncc_cmc = ncc_cmc+cumsum(inds==query_label);
end
ncc_cmc = ncc_cmc./300.*100;


baselines = {'datasets/FID-300/result_ACCV14.mat', ...
             'datasets/FID-300/ranks_BMVC16.mat', ...
             'datasets/FID-300/ranks_LoG16.mat'};
base_cmc = zeros(numel(baselines), 1175);
for b=1:numel(baselines)
  load(baselines{b}, 'ranks')

  for p=1:300
    res = zeros(1, 1175);
    res(ranks(p)) = 1;
    base_cmc(b, :) = base_cmc(b, :)+cumsum(res);
  end
  base_cmc(b, :) = base_cmc(b, :)./300.*100;
end


set(0,'DefaultAxesFontName', 'Times New Roman')
set(0,'DefaultTextFontname', 'Times New Roman')
font_size = 48;
line_width = 6;

colors = get(gca, 'colororder');
hold on
for b=1:numel(baselines)
  plot([1:1175]./1175.*100, base_cmc(b, :), ...
       'LineWidth', line_width, ...
       'Color', colors(b, :))
end
plot([1:1175]./1175.*100, ncc_cmc{1}, ...
     'LineWidth', line_width, ...
     'Color', colors(numel(baselines)+1, :))
hold off
grid on
xlim([0 10]), ylim([0 100])
xlabel('# database images reviewed (%)')
ylabel('# correct matches (%)')
lgd = legend('ACCV14', 'BMVC16', 'LoG16', ...
             '$[\mu_c,\sigma_c]$', ...
             'Location', 'NorthEastOutside');
lgd.Interpreter = 'latex';
axis square
set(gca, 'FontSize', font_size)
set(findall(gcf,'type','text'), 'FontSize', font_size)


end
