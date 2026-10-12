#!/usr/bin/env python3
import socket
import struct
import sys
import time
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Send CSV via UDP in protocol v1")
    parser.add_argument("file", help="CSV file with t,x,y[,valid]")
    parser.add_argument("--host", default="127.0.0.1", help="Target host")
    parser.add_argument("--port", type=int, default=5005, help="Target port")
    parser.add_argument("--speed", type=float, default=1.0, help="Replay speed (0 for ASAP)")
    parser.add_argument("--batch", type=int, default=1, help="Samples per packet")
    return parser.parse_args()

def main():
    args = parse_args()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    MAGIC = 0x31455945

    start_time = None
    first_t = None

    batch = []

    def send_batch():
        if not batch: return
        n = len(batch)
        data = struct.pack("<II", MAGIC, n)
        for s in batch:
            data += struct.pack("<dffI", s[0], s[1], s[2], s[3])
        sock.sendto(data, (args.host, args.port))
        batch.clear()

    with open(args.file, 'r') as f:
        # Check header
        header = f.readline().strip()
        if not header:
            return
        if header[0] not in '+-.0123456789':
            pass # skipped
        else:
            f.seek(0)

        for line in f:
            line = line.replace(',', ' ').strip()
            if not line: continue
            parts = line.split()
            if len(parts) >= 3:
                t = float(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                flags = 1
                if len(parts) >= 4:
                    if float(parts[3]) == 0:
                        flags = 0

                if first_t is None:
                    first_t = t
                    start_time = time.time()
                elif args.speed > 0:
                    now = time.time()
                    elapsed_real = now - start_time
                    elapsed_sim = (t - first_t) / args.speed
                    if elapsed_sim > elapsed_real:
                        time.sleep(elapsed_sim - elapsed_real)

                batch.append((t, x, y, flags))
                if len(batch) >= args.batch:
                    send_batch()

    send_batch()

if __name__ == "__main__":
    main()
