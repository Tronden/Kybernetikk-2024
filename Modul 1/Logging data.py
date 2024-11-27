import serial
import time

# Specify the serial port and the baud rate
ser = serial.Serial('COM4', 115200, timeout=1)  # Update 'COM4' to the correct port
time.sleep(2)  # Allow time for the serial connection to initialize

# Generate a unique filename based on the current date and time
filename = time.strftime("motor_data_log_%H-%M-%S_%d-%m-%Y.txt")

# Open the file in append mode
with open(filename, "a") as file:
    print(f"Logging data from Arduino to {filename}... Press Ctrl+C to stop.")
    try:
        while True:
            # Read a line from the serial port
            line = ser.readline().decode('utf-8').strip()
            
            if line:  # Check if there's valid data
                # Get the current time with milliseconds
        

                # Create the log entry with timestamp and data
                log_entry = f"{line}"
                
                # Print the log entry to the console (optional)
                print(log_entry)
                
                # Write the log entry to the file
                file.write(log_entry + '\n')
                
                # Flush the file to ensure data is written
                file.flush()
    
    except KeyboardInterrupt:
        print("Logging stopped.")
    
    finally:
        ser.close()  # Close the serial connection
