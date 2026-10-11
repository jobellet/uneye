% EYELIVE_FOLLOW_EXAMPLE A basic script showing a Psychtoolbox loop
% reading from EyeLive in real time.

% Open the mapped file
mm = eyelive_open('session.bin');

disp('Press Ctrl+C to stop following...');

last_event = 0;

while true
    % Read the latest state
    [state, ok] = eyelive_state(mm);

    if ok
        % Check for new events
        if state.event_counter > last_event
            fprintf('Event: %d | Cls: %d | Amp: %.2f deg | Peak speed: %.1f deg/s\n', ...
                state.event_counter, state.ev_cls, state.ev_amplitude_deg, state.ev_peak_speed_deg_s);
            last_event = state.event_counter;
        end

        % In a real PTB loop, you might draw gaze position:
        % Screen('DrawDots', window, [state.x; state.y], size, color, [], 1);
        % Screen('Flip', window);
    end

    pause(0.005);
end
