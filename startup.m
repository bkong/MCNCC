addpath('utils')

if ~exist(fullfile('utils', ['hashMat.', mexext]), 'file')
  cd 'utils'
  mex hashMat.cc -O -lssl -lcrypto
  cd '..'
end

run(fullfile(fileparts(mfilename('fullpath')),...
  'matconvnet', 'matlab', 'vl_setupnn.m'));
