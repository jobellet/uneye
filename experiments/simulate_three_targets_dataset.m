function varargout = simulate_three_targets_dataset(varargin)
% SIMULATE_THREE_TARGETS_DATASET Wrapper to launch dataset simulation.
    here = fileparts(mfilename('fullpath'));
    matlab_dir = fullfile(here, '..', 'cpp', 'app', 'readers', 'matlab');
    addpath(matlab_dir);
    [varargout{1:nargout}] = simulate_three_targets_dataset(varargin{:});
end
