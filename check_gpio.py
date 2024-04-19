import Jetson.GPIO as GPIO

def main(): 
    GPIO.setmode(GPIO.BOARD)
    GPIO.setwarnings(False)

    print("GPIO Ready to use.")

    try:
        while True: 
            pass

    finally:
        GPIO.cleanup()
        print("Bye.")

if __name__ == "__main__":
    main()