#!/usr/bin/env python3
"""
Scan available serial ports and probe each for the thermal camera response.

Usage:
  python3 scripts/thermal_autoscan.py

This script requires pyserial (for list_ports). It will try a set of common
baud rates on each detected port and report any ports that return data or the
expected header 0x5A5A.
"""
import time
import binascii
from serial.tools import list_ports
import subprocess
import sys

DEFAULT_BAUDS = [115200, 230400, 460800, 921600]
COMMAND_READ_FRAME = b'\xA5\x45\xEA'
EXPECTED_HEADER = b'\x5A\x5A'

def hexdump(b: bytes, length=256):
    s = binascii.hexlify(b[:length]).decode('ascii')
    return ' '.join([s[i:i+2] for i in range(0, len(s), 2)])

def probe_port(port, bauds=DEFAULT_BAUDS, read_bytes=1024, tries=1):
    import serial
    results = []
    for baud in bauds:
        try:
            ser = serial.Serial(port.device, baud, timeout=1)
        except Exception as e:
            results.append((baud, 0, False, f"open-failed: {e}"))
            continue

        successful = False
        total_read = 0
        for t in range(tries):
            try:
                ser.reset_input_buffer()
            except Exception:
                pass
            ser.write(COMMAND_READ_FRAME)
            time.sleep(0.1)
            data = ser.read(read_bytes)
            total_read += len(data)
            if EXPECTED_HEADER in data:
                successful = True
        try:
            ser.close()
        except Exception:
            pass

        results.append((baud, total_read, successful, hexdump(data, length=64) if data else ''))

    return results

def main():
    ports = list(list_ports.comports())
    if not ports:
        print("No serial ports detected. Make sure your device is connected.")
        sys.exit(1)

    print(f"Detected ports: {[p.device for p in ports]}")
    for p in ports:
        print(f"\nProbing {p.device} ({p.description})")
        res = probe_port(p, tries=2)
        for baud, total, ok, dump in res:
            status = 'HEADER' if ok else ('DATA' if total>0 else 'NONE')
            print(f"  {baud} -> {status}, bytes={total}")
            if dump:
                print(f"    sample: {dump}")

if __name__ == '__main__':
    main()
