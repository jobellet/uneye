import sys
import struct

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 test_reader.py <filename>")
        sys.exit(1)

    filename = sys.argv[1]

    with open(filename, 'rb') as f:
        data = f.read(256)

    if len(data) != 256:
        print("Error: Block size is not 256")
        sys.exit(1)

    # Read some specific values to verify
    # rate_hz is at offset 156 (double) as defined by C++ struct layout
    rate_hz = struct.unpack_from('<d', data, 156)[0]
    if rate_hz != 120.0:
        print(f"Error: expected rate_hz=120.0, got {rate_hz}")
        sys.exit(1)

    # x is at offset 24 (float)
    x = struct.unpack_from('<f', data, 24)[0]
    if x != 5.0:
        print(f"Error: expected x=5.0, got {x}")
        sys.exit(1)

    print("Python test script verified block successfully.")

if __name__ == '__main__':
    main()
