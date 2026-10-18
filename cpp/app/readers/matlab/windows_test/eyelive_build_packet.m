function bytes = eyelive_build_packet(t, x, y, flags)
% EYELIVE_BUILD_PACKET  One binary datagram of the EyeLive UDP protocol v1 (cpp/app/include/eyelive/source.hpp).
%   bytes = eyelive_build_packet(t, x, y)   t in seconds (double), x, y in the units of the profile (single), up to 50 samples
% Layout (little endian): uint32 magic 0x31455945 ('EYE1'), uint32 count, then count x { double t, single x, single y, uint32 flags } (20 bytes per sample).
    n = numel(t); if n < 1 || n > 50, error('1 to 50 samples per packet'); end
    if nargin < 4, flags = ones(n, 1, 'uint32'); end
    bytes = zeros(1, 8 + 20 * n, 'uint8'); bytes(1:8) = typecast(uint32([hex2dec('31455945') n]), 'uint8');
    for k = 1:n
        o = 8 + 20 * (k - 1);
        bytes(o + (1:8)) = typecast(double(t(k)), 'uint8'); bytes(o + (9:12)) = typecast(single(x(k)), 'uint8'); bytes(o + (13:16)) = typecast(single(y(k)), 'uint8'); bytes(o + (17:20)) = typecast(uint32(flags(k)), 'uint8');
    end
end
