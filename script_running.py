import RPi.GPIO as GPIO
import subprocess
import time

# ==============================================================
# CONFIGURATION
# ==============================================================
# GPIO pin numbers
BUTTON_1_PIN = 17
BUTTON_2_PIN = 27

# Virtual environment path
VENV_PATH = "/home/pi/myenv/bin/activate"   # <-- change to your venv path

# Scripts to run
SCRIPT_1 = "/home/pi/scripts/script1.py"    # <-- change to your script path
SCRIPT_2 = "/home/pi/scripts/script2.py"

# ==============================================================
# SETUP GPIO
# ==============================================================
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_1_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(BUTTON_2_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

print("✅ System ready. Press Button 1 or Button 2 to run respective scripts...")

def run_in_venv(script_path):
    """Run a Python script inside the virtual environment."""
    try:
        # The command activates venv and runs the script
        command = f"bash -c 'source {VENV_PATH} && python {script_path}'"
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {script_path}: {e}")

# ==============================================================
# MAIN LOOP
# ==============================================================
try:
    while True:
        # Button 1 pressed
        if GPIO.input(BUTTON_1_PIN) == GPIO.LOW:
            print("▶ Running Script 1...")
            run_in_venv(SCRIPT_1)
            time.sleep(1)  # debounce delay

        # Button 2 pressed
        if GPIO.input(BUTTON_2_PIN) == GPIO.LOW:
            print("▶ Running Script 2...")
            run_in_venv(SCRIPT_2)
            time.sleep(1)

        time.sleep(0.1)

except KeyboardInterrupt:
    print("\n🛑 Exiting...")
finally:
    GPIO.cleanup()
    print("GPIO cleaned up. Bye!")
