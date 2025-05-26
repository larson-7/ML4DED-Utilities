import time
import bambulabs_api as bl

IP = "192.168.1.171"
SERIAL = "01P09A451701831"
ACCESS_CODE = "25766094"

if __name__ == "__main__":
    print("Starting bambulabs_api example")
    print("Connecting to Bambulabs 3D printer")
    print(f"IP: {IP}")
    print(f"Serial: {SERIAL}")
    print(f"Access Code: {ACCESS_CODE}")

    # Create a new instance of the API
    printer = bl.Printer(IP, ACCESS_CODE, SERIAL)

    # Connect to the Bambulabs 3D printer
    printer.connect()

    time.sleep(2)
    printer.turn_light_on()
    time.sleep(2)

    # Get the printer status
    status = printer.get_state()
    print(f"Printer status: {status}")
    image = printer.get_camera_image()
    image.save("example.png")
    # Turn the light off
    printer.turn_light_off()

    time.sleep(2)

    # Turn the light on
    printer.turn_light_on()

    # Disconnect from the Bambulabs 3D printer
    printer.disconnect()
