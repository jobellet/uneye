function [state, success] = eyelive_state(mm, max_retries)
% EYELIVE_STATE Reads a consistent state from the memmapfile object.
%
% [state, success] = eyelive_state(mm, [max_retries=100])
% Returns the state struct and true if successful, or false if it failed
% after max_retries (e.g. writer held the lock for too long).

    if nargin < 2
        max_retries = 100;
    end

    success = false;
    state = struct();

    for i = 1:max_retries
        % 1. Read first sequence
        seq1 = mm.Data.seq;

        % If odd, writer is writing
        if bitand(seq1, 1) ~= 0
            pause(0.001); % yield
            continue;
        end

        % 2. Copy the entire state data
        raw_state = mm.Data;

        % 3. Read second sequence
        seq2 = mm.Data.seq;

        % If they match, copy is consistent
        if seq1 == seq2
            state = raw_state;
            success = true;
            return;
        end
    end
end
