import sys

RED = "\033[31m"
RESET = "\033[0m"

def run_demo_module():
    try:
        from ai_trader.demo.demo_module import run_demo
        run_demo()
    except ImportError as e:
        print(f"Error while importing demo module: {e}")
    except Exception as e:
        print(f"Error while running demo module: {e}")

def main():
    while True:
        print(f"\n{RED}AI-Trader Main Menu{RESET}\n")
        print("1. Launch demo module")
        print("2. Exit\n")
        
        choice = input("Choose an option: ").strip()
        
        if choice == '1':
            print()
            run_demo_module()
        elif choice == '2':
            print("\nExiting.\n")
            sys.exit(0)
        else:
            print("Wrong choice, please try again.")


if __name__ == "__main__":
    main()

