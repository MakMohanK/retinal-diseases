from gpiozero import Button
from time import sleep

# Set pull_up=False to enable internal pull-down resistor, since button connect>
button1 = Button(23, pull_up=False)
button2 = Button(24, pull_up=False)

while True:
    if button1.is_pressed:
        print("Button 1 (GPIO 23) is pressed")
    if button2.is_pressed:
        print("Button 2 (GPIO 24) is pressed")
    sleep(0.1)
