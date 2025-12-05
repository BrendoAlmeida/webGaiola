#!/usr/bin/env python3
"""
Simple diagnostic probe for the thermal camera serial output.

Usage:
  python3 scripts/thermal_probe.py --port /dev/ttyS0

This script will try a list of common baud rates, send the
COMMAND_READ_FRAME (0xA5 0x45 0xEA), and read back up to N bytes.
It prints whether the expected header 0x5A5A appears and a hex
dump of the first bytes to help debugging.
"""
import argparse
import serial
import time
import binascii

COMMAND_READ_FRAME = b'\xA5\x45\xEA'
EXPECTED_HEADER = b'\x5A\x5A'

DEFAULT_BAUDS = [115200, 230400, 460800, 921600]

def hexdump(b: bytes, length=256):
    s = binascii.hexlify(b[:length]).decode('ascii')
    # group by two chars
    return ' '.join([s[i:i+2] for i in range(0, len(s), 2)])

def probe_port(port, bauds=None, read_bytes=2048, tries=1, pause=0.1):
    bauds = bauds or DEFAULT_BAUDS
    for baud in bauds:
        print(f"\n--- Trying {port} at {baud} bps ---")
        try:
            ser = serial.Serial(port, baud, timeout=1)
        except Exception as e:
            print(f"Failed to open {port} at {baud}: {e}")
            continue

        try:
            for t in range(tries):
                print(f"Sending frame request (try {t+1}/{tries})...")
                try:
                    ser.reset_input_buffer()
                except Exception:
                    pass
                ser.write(COMMAND_READ_FRAME)
                time.sleep(pause)
                data = ser.read(read_bytes)
                print(f"Read {len(data)} bytes")
                if EXPECTED_HEADER in data:
                    idx = data.find(EXPECTED_HEADER)
                    print(f"Found header 0x5A5A at offset {idx}")
                else:
                    print("Header 0x5A5A not found in response.")

                if data:
                    print("Hexdump (first 512 bytes):")
                    print(hexdump(data, length=512))
                else:
                    print("No data read from device.")

            ser.close()
        except Exception as exc:
            print(f"Error during probe at {baud}: {exc}")
            try:
                ser.close()
            except Exception:
                pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', required=True, help='Serial port to probe (e.g., /dev/ttyS0 or COM3)')
    parser.add_argument('--bauds', nargs='+', type=int, help='List of baud rates to try')
    parser.add_argument('--bytes', type=int, default=2048, help='Number of bytes to read')
    parser.add_argument('--tries', type=int, default=1, help='Number of request/read cycles per baud')
    args = parser.parse_args()

    probe_port(args.port, bauds=args.bauds, read_bytes=args.bytes, tries=args.tries)

if __name__ == '__main__':
    main()
