import time
import numpy as np
import cv2
import serial

# --- Camera and Recording Constants ---
FRAME_WIDTH = 32
FRAME_HEIGHT = 24
TOTAL_PIXELS = FRAME_WIDTH * FRAME_HEIGHT
COMMAND_READ_FRAME = b'\xA5\x45\xEA'
HEADER = b'\x5A\x5A'

# --- Recording Settings ---
RECORDING_DURATION_SECONDS = 60
OUTPUT_FILENAME = 'thermal_recording.avi'
OUTPUT_RESOLUTION = (640, 480)
FOURCC = cv2.VideoWriter_fourcc(*'MJPG')
FPS = 10

def parse_frame(data):
    """Extracts and validates temperature data from the serial packet."""
    if not data or len(data) < 1544:
        print("Invalid data packet received.")
        return None

    temp_data = data[4:4 + (TOTAL_PIXELS * 2)]
    if len(temp_data) != TOTAL_PIXELS * 2:
        print("Temperature data has incorrect length.")
        return None

    temps_raw = np.frombuffer(temp_data, dtype=np.int16)
    temps_celsius = temps_raw / 100.0

    return temps_celsius.reshape((FRAME_HEIGHT, FRAME_WIDTH))

def process_thermal_frame(ser):
    """
    Requests, reads, and processes a frame from the thermal camera robustly,
    synchronizing with the packet header to avoid corrupted data.
    """
    try:
        ser.write(COMMAND_READ_FRAME)

        while True:
            byte1 = ser.read(1)
            if not byte1:
                return None # Timeout

            if byte1 == HEADER[0:1]:
                byte2 = ser.read(1)
                if not byte2:
                    return None

                if byte2 == HEADER[1:2]:
                    break

        remaining_bytes = ser.read(1544 - 2)
        if len(remaining_bytes) < (1544 - 2):
            print("Incomplete packet received after header.")
            return None

        data_packet = HEADER + remaining_bytes
        temperatures = parse_frame(data_packet)

        if temperatures is None:
            print("Failed to parse a seemingly valid packet.")
            return None

        min_temp, max_temp = np.min(temperatures), np.max(temperatures)

        normalized_frame = cv2.normalize(temperatures, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        heatmap_img = cv2.applyColorMap(normalized_frame, cv2.COLORMAP_INFERNO)

        resized_heatmap = cv2.resize(heatmap_img, OUTPUT_RESOLUTION, interpolation=cv2.INTER_CUBIC)

        info_text = f"Min: {min_temp:.1f}C Max: {max_temp:.1f}C"
        cv2.putText(resized_heatmap, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return resized_heatmap

    except Exception as e:
        print(f"Error in thermal camera processing: {e}")
        if ser and ser.is_open:
            ser.reset_input_buffer()
        return None

def record_thermal_video(serial_port_name, baud_rate=460800):
    """
    Connects to the thermal camera and records a video for a specified duration.
    """
    ser = None
    video_writer = None
    try:
        print(f"Connecting to serial port {serial_port_name} at {baud_rate}...")
        ser = serial.Serial(serial_port_name, baud_rate, timeout=1)
        ser.reset_input_buffer()
        print("Serial port connected and input buffer cleared.")

        video_writer = cv2.VideoWriter(OUTPUT_FILENAME, FOURCC, FPS, OUTPUT_RESOLUTION)
        if not video_writer.isOpened():
            print("Error: Could not open video writer. Check codec and permissions.")
            return
        print(f"Recording started. Output file: {OUTPUT_FILENAME}")

        start_time = time.time()
        while (time.time() - start_time) < RECORDING_DURATION_SECONDS:
            frame = process_thermal_frame(ser)

            if frame is not None:
                video_writer.write(frame)


            else:
                time.sleep(0.1)

        elapsed_time = time.time() - start_time
        print(f"Recording finished. Total duration: {elapsed_time:.2f} seconds.")

    except serial.SerialException as e:
        print(f"Serial Error: {e}. Check port name and device connection.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        print("Cleaning up resources.")
        if ser and ser.is_open:
            ser.close()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    SERIAL_PORT = '/dev/ttyS0'
    record_thermal_video(SERIAL_PORT, 115200)
