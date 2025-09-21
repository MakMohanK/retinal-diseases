from gpiozero import LED
from time import sleep

led = LED(4)

while True:
    led.on()
    print("LED is ON")
    sleep(1)
    led.off()
    print("LED is OFF")
    sleep(1)

