import mmap
import struct
import os
import time

class StateReader:
    def __init__(self, filename):
        self.filename = filename
        self.fd = os.open(filename, os.O_RDONLY)
        # Map only the first 256 bytes for state block
        # (Assuming the file is larger, we just need 512, state block is [256, 512))
        self.mmap = mmap.mmap(self.fd, 512, mmap.MAP_SHARED, mmap.PROT_READ)

        # Format string for struct unpacking, matching the 256-byte StateBlock
        # uint32 magic, uint32 version, uint64 seq, double t_last, float x, y, float speed
        # uint8 cls, in_event, uint16 n_classes, uint64 n_samples, uint64 ev_counter, uint64 prov_counter
        # double ev_t_onset, double ev_t_offset, float ev_amp, ev_peak, ev_dir, uint8 ev_cls, ev_prov, uint16 ev_pad, uint64 ev_id
        # float pred_x, pred_y, pred_sx, pred_sy, double pred_t_valid, uint8 pred_valid, pad[3]
        # uint32 health, uint32 dropped, float lat_p50, lat_p95, double deg_per, double rate

        # We define a helper to unpack directly from offsets rather than full struct format to be robust
        # Offsets (hardcoded from C++ struct layout)
        self.OFFSETS = {
            'magic': 0, 'version': 4, 'seq': 8, 't_last': 16,
            'x': 24, 'y': 28, 'speed_deg_s': 32, 'cls': 36, 'in_event': 37,
            'n_classes': 38, 'n_samples': 40, 'event_counter': 48,
            'provisional_counter': 56, 'ev_t_onset': 64, 'ev_t_offset': 72,
            'ev_amplitude_deg': 80, 'ev_peak_speed_deg_s': 84, 'ev_direction_rad': 88,
            'ev_cls': 92, 'ev_provisional': 93, 'ev_pad': 94, 'ev_id': 96,
            'pred_x': 104, 'pred_y': 108, 'pred_sigma_x': 112, 'pred_sigma_y': 116,
            'pred_t_valid': 120, 'pred_valid': 128, 'pred_pad': 129,
            'health_flags': 132, 'dropped_packets': 136,
            'detect_latency_ms_p50': 140, 'detect_latency_ms_p95': 144,
            'deg_per_unit': 148, 'rate_hz': 156
        }

    def __del__(self):
        self.mmap.close()
        os.close(self.fd)

    def latest(self, max_retries=100):
        # Read from offset 256
        base = 256
        for _ in range(max_retries):
            # Read seq1
            seq1 = struct.unpack_from('<Q', self.mmap, base + self.OFFSETS['seq'])[0]
            if (seq1 & 1) != 0:
                time.sleep(0) # yield
                continue

            # Copy memory block (Python read is essentially a copy)
            block_data = self.mmap[base:base+256]

            # Read seq2
            seq2 = struct.unpack_from('<Q', self.mmap, base + self.OFFSETS['seq'])[0]

            if seq1 == seq2:
                # Successfully read a consistent block, parse fields
                state = {}
                state['seq'] = seq1
                state['magic'] = struct.unpack_from('<I', block_data, self.OFFSETS['magic'])[0]
                state['version'] = struct.unpack_from('<I', block_data, self.OFFSETS['version'])[0]
                state['t_last'] = struct.unpack_from('<d', block_data, self.OFFSETS['t_last'])[0]
                state['x'] = struct.unpack_from('<f', block_data, self.OFFSETS['x'])[0]
                state['y'] = struct.unpack_from('<f', block_data, self.OFFSETS['y'])[0]
                state['speed_deg_s'] = struct.unpack_from('<f', block_data, self.OFFSETS['speed_deg_s'])[0]
                state['cls'] = struct.unpack_from('<B', block_data, self.OFFSETS['cls'])[0]
                state['in_event'] = struct.unpack_from('<B', block_data, self.OFFSETS['in_event'])[0]
                state['n_classes'] = struct.unpack_from('<H', block_data, self.OFFSETS['n_classes'])[0]
                state['n_samples'] = struct.unpack_from('<Q', block_data, self.OFFSETS['n_samples'])[0]
                state['event_counter'] = struct.unpack_from('<Q', block_data, self.OFFSETS['event_counter'])[0]
                state['provisional_counter'] = struct.unpack_from('<Q', block_data, self.OFFSETS['provisional_counter'])[0]
                state['ev_t_onset'] = struct.unpack_from('<d', block_data, self.OFFSETS['ev_t_onset'])[0]
                state['ev_t_offset'] = struct.unpack_from('<d', block_data, self.OFFSETS['ev_t_offset'])[0]
                state['ev_amplitude_deg'] = struct.unpack_from('<f', block_data, self.OFFSETS['ev_amplitude_deg'])[0]
                state['ev_peak_speed_deg_s'] = struct.unpack_from('<f', block_data, self.OFFSETS['ev_peak_speed_deg_s'])[0]
                state['ev_direction_rad'] = struct.unpack_from('<f', block_data, self.OFFSETS['ev_direction_rad'])[0]
                state['ev_cls'] = struct.unpack_from('<B', block_data, self.OFFSETS['ev_cls'])[0]
                state['ev_provisional'] = struct.unpack_from('<B', block_data, self.OFFSETS['ev_provisional'])[0]
                state['ev_id'] = struct.unpack_from('<Q', block_data, self.OFFSETS['ev_id'])[0]
                state['pred_x'] = struct.unpack_from('<f', block_data, self.OFFSETS['pred_x'])[0]
                state['pred_y'] = struct.unpack_from('<f', block_data, self.OFFSETS['pred_y'])[0]
                state['pred_sigma_x'] = struct.unpack_from('<f', block_data, self.OFFSETS['pred_sigma_x'])[0]
                state['pred_sigma_y'] = struct.unpack_from('<f', block_data, self.OFFSETS['pred_sigma_y'])[0]
                state['pred_t_valid'] = struct.unpack_from('<d', block_data, self.OFFSETS['pred_t_valid'])[0]
                state['pred_valid'] = struct.unpack_from('<B', block_data, self.OFFSETS['pred_valid'])[0]
                state['health_flags'] = struct.unpack_from('<I', block_data, self.OFFSETS['health_flags'])[0]
                state['dropped_packets'] = struct.unpack_from('<I', block_data, self.OFFSETS['dropped_packets'])[0]
                state['detect_latency_ms_p50'] = struct.unpack_from('<f', block_data, self.OFFSETS['detect_latency_ms_p50'])[0]
                state['detect_latency_ms_p95'] = struct.unpack_from('<f', block_data, self.OFFSETS['detect_latency_ms_p95'])[0]
                state['deg_per_unit'] = struct.unpack_from('<d', block_data, self.OFFSETS['deg_per_unit'])[0]
                state['rate_hz'] = struct.unpack_from('<d', block_data, self.OFFSETS['rate_hz'])[0]

                return state
        return None

    def follow_events(self):
        # Generator that continuously yields new final events
        last_event_counter = 0
        while True:
            state = self.latest()
            if state is not None and state['event_counter'] > last_event_counter:
                last_event_counter = state['event_counter']
                yield {
                    't_onset': state['ev_t_onset'],
                    't_offset': state['ev_t_offset'],
                    'amplitude_deg': state['ev_amplitude_deg'],
                    'peak_speed_deg_s': state['ev_peak_speed_deg_s'],
                    'direction_rad': state['ev_direction_rad'],
                    'cls': state['ev_cls'],
                    'id': state['ev_id']
                }
            time.sleep(0.001)
