function varargout = ptb_three_targets_game(varargin)
% PTB_THREE_TARGETS_GAME Wrapper to launch the PTB Gaze Prediction Game.
    here = fileparts(mfilename('fullpath'));
    matlab_dir = fullfile(here, '..', 'cpp', 'app', 'readers', 'matlab');
    addpath(matlab_dir);
    [varargout{1:nargout}] = ptb_three_targets_game(varargin{:});
end
