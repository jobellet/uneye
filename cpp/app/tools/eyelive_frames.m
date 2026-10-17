% EyeLive frame mailbox writer for Psychtoolbox on MATLAB or Octave. NOT YET RUN: written from the file format (cpp/app/engine/include/eyelive/frame_mailbox.hpp); the C++ and Python writers are tested.
% Octave has no memmapfile: there the file is written with fopen / fseek / fwrite / fflush (the engine's memory map sees the page cache); MATLAB uses memmapfile.
%
%   fw = eyelive_frames('open', '/tmp/stim.frames', 64, 64, 3, [xl yt xr yb]);   % gaze units of the frame edges (left, top, right, bottom)
%   img = imresize(Screen('GetImage', smallWin, [], 'backBuffer'), [64 64]);       % e.g. a 64 x 64 off-screen copy of the stimulus (uint8, h x w x 3)
%   fw = eyelive_frames('frame', fw, img, GetSecs);
%   fw = eyelive_frames('candidates', fw, [x1 y1 w1; x2 y2 w2], 12, GetSecs);       % up to 64 rows, gaze units, weight > 0; then use_until_ms and the time stamp (same clock as the frames)
%   eyelive_frames('close', fw);
function fw = eyelive_frames(cmd, fw, varargin)
  slots = 3; HEADER = 256; MAXC = 64; SLOTH = 32;
  isoct = exist('OCTAVE_VERSION', 'builtin') ~= 0;
  switch cmd
    case 'open'
      path = fw; w = varargin{1}; h = varargin{2}; c = varargin{3}; rect = varargin{4};
      slotbytes = SLOTH + w*h*c; total = HEADER + 16*MAXC + slotbytes*slots;
      f = fopen(path, 'w', 'ieee-le'); fwrite(f, zeros(total, 1, 'uint8'), 'uint8'); fclose(f);
      fw = struct('isoct', isoct, 'n', uint64(0), 'slots', slots, 'w', w, 'h', h, 'c', c, 'slotbytes', slotbytes, 'fid', -1, 'm', []);
      if isoct
        fw.fid = fopen(path, 'r+', 'ieee-le');
      else
        fw.m = memmapfile(path, 'Format', {'uint8', [total 1], 'raw'}, 'Writable', true);
      end
      put(fw, 0, uint8('EYLFRM01')'); put(fw, 8, uint32([1 HEADER w h c slots SLOTH MAXC])); put(fw, 48, double(rect(:))'); flush_(fw);
    case 'frame'
      img = varargin{1}; t = varargin{2}; fw.n = fw.n + 1; k = double(mod(fw.n - 1, fw.slots));
      off = HEADER + 16*MAXC + fw.slotbytes * k; seq = get64(fw, off);
      put(fw, off, uint64(seq + 1)); flush_(fw);                                                       % odd: being written
      put(fw, off + 8, double(t)); put(fw, off + 16, uint64(fw.n)); put(fw, off + 24, uint32([0 0]));
      put(fw, off + SLOTH, reshape(permute(uint8(img), [3 2 1]), 1, []));                              % h x w x c (column-major) -> c x w x h = row-major interleaved
      flush_(fw); put(fw, off, uint64(seq + 2)); flush_(fw);                                           % even: done
      put(fw, 40, uint64(fw.n)); flush_(fw);                                                           % published last
    case 'candidates'
      cand = varargin{1}; until_ms = 1e9; if numel(varargin) > 1, until_ms = varargin{2}; end
      tcand = 0; if numel(varargin) > 2, tcand = varargin{3}; end                                                  % time stamp in the writer's clock (same clock as the frames, e.g. GetSecs), 0 = none
      seq = get64(fw, 80); put(fw, 80, uint64(seq + 1)); flush_(fw); n = min(size(cand, 1), 64); put(fw, 88, uint32(n)); put(fw, 96, double(until_ms)); put(fw, 104, double(tcand));
      blk = zeros(4, 64, 'single'); if n > 0, blk(1:2, 1:n) = single(cand(1:n, 1:2)'); if size(cand, 2) >= 3, blk(3, 1:n) = single(cand(1:n, 3)'); else, blk(3, 1:n) = 1; end, end
      put(fw, HEADER, reshape(blk, 1, [])); flush_(fw); put(fw, 80, uint64(seq + 2)); flush_(fw);
    case 'close'
      if fw.isoct, fclose(fw.fid); end
      clear fw;
  end
end

function put(fw, off, v)                       % write the array v at byte offset off (little endian, native class of v)
  if fw.isoct, fseek(fw.fid, off, 'bof'); fwrite(fw.fid, v, class(v));
  else, bytes = typecast(v(:)', 'uint8'); fw.m.Data.raw(off + 1:off + numel(bytes)) = bytes; end
end
function s = get64(fw, off)
  if fw.isoct, fseek(fw.fid, off, 'bof'); s = fread(fw.fid, 1, 'uint64=>uint64');
  else, s = typecast(fw.m.Data.raw(off + 1:off + 8), 'uint64'); end
end
function flush_(fw)
  if fw.isoct, fflush(fw.fid); end
end
