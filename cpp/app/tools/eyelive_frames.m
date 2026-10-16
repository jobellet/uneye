% EyeLive frame mailbox writer for Psychtoolbox / MATLAB / Octave (NOT TESTED: written from the file format only; the C++ and Python writers are tested).
% Layout: cpp/app/engine/include/eyelive/frame_mailbox.hpp.
%
%   fw = eyelive_frames('open', '/tmp/stim.frames', 64, 64, 3, [xl yt xr yb]);   % gaze units of the frame edges (left, top, right, bottom)
%   img = imresize(Screen('GetImage', smallWin, [], 'backBuffer'), [64 64]);       % e.g. a 64 x 64 off-screen copy of the stimulus (uint8, h x w x 3)
%   fw = eyelive_frames('frame', fw, img, GetSecs);
%   fw = eyelive_frames('candidates', fw, [x1 y1 w1; x2 y2 w2], 12);               % up to 64 rows, gaze units, weight > 0; last argument: use_until_ms
%   eyelive_frames('close', fw);
function fw = eyelive_frames(cmd, fw, varargin)
  slots = 3; HEADER = 256; MAXC = 64; SLOTH = 32;
  switch cmd
    case 'open'
      path = fw; w = varargin{1}; h = varargin{2}; c = varargin{3}; rect = varargin{4};
      slotbytes = SLOTH + w*h*c; total = HEADER + 16*MAXC + slotbytes*slots;
      f = fopen(path, 'w'); fwrite(f, zeros(total, 1, 'uint8'), 'uint8'); fclose(f);
      fmt = {'uint8', [8 1], 'magic'; 'uint32', [1 1], 'version'; 'uint32', [1 1], 'header'; 'uint32', [1 1], 'w'; 'uint32', [1 1], 'h'; 'uint32', [1 1], 'c'; 'uint32', [1 1], 'slots'; ...
             'uint32', [1 1], 'slothdr'; 'uint32', [1 1], 'maxc'; 'uint64', [1 1], 'count'; 'double', [4 1], 'rect'; 'uint64', [1 1], 'cseq'; 'uint32', [1 1], 'cn'; 'uint32', [1 1], 'pad'; ...
             'double', [1 1], 'cuntil'; 'uint8', [152 1], 'reserved'; 'single', [4 MAXC], 'cand'};
      for k = 1:slots
        fmt = [fmt; {'uint64', [1 1], sprintf('seq%d', k); 'double', [1 1], sprintf('t%d', k); 'uint64', [1 1], sprintf('no%d', k); 'uint32', [1 2], sprintf('fl%d', k); 'uint8', [c w h], sprintf('px%d', k)}]; %#ok<AGROW>
      end
      m = memmapfile(path, 'Format', fmt, 'Writable', true);
      m.Data.magic = uint8('EYLFRM01')'; m.Data.version = uint32(1); m.Data.header = uint32(HEADER); m.Data.w = uint32(w); m.Data.h = uint32(h); m.Data.c = uint32(c); m.Data.slots = uint32(slots);
      m.Data.slothdr = uint32(SLOTH); m.Data.maxc = uint32(MAXC); m.Data.rect = double(rect(:));
      fw = struct('m', m, 'n', uint64(0), 'slots', slots, 'w', w, 'h', h, 'c', c);
    case 'frame'
      img = varargin{1}; t = varargin{2}; fw.n = fw.n + 1; k = double(mod(fw.n - 1, fw.slots)) + 1;
      sq = sprintf('seq%d', k); seq = fw.m.Data.(sq);
      fw.m.Data.(sq) = seq + 1;                                           % odd: being written
      fw.m.Data.(sprintf('t%d', k)) = double(t); fw.m.Data.(sprintf('no%d', k)) = fw.n;
      fw.m.Data.(sprintf('px%d', k)) = permute(uint8(img), [3 2 1]);       % h x w x c (column-major) -> c x w x h = row-major interleaved
      fw.m.Data.(sq) = seq + 2;                                            % even: done
      fw.m.Data.count = fw.n;                                              % published last
    case 'candidates'
      cand = varargin{1}; until = 1e9; if numel(varargin) > 1, until = varargin{2}; end
      seq = fw.m.Data.cseq; fw.m.Data.cseq = seq + 1; n = min(size(cand, 1), 64); fw.m.Data.cn = uint32(n); fw.m.Data.cuntil = double(until);
      blk = zeros(4, 64, 'single'); if n > 0, blk(1:2, 1:n) = single(cand(1:n, 1:2)'); if size(cand, 2) >= 3, blk(3, 1:n) = single(cand(1:n, 3)'); else, blk(3, 1:n) = 1; end, end
      fw.m.Data.cand = blk; fw.m.Data.cseq = seq + 2;
    case 'close'
      clear fw;
  end
end
