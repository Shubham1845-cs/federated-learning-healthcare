import subprocess
import time
import sys
import os

class Colors:
    """ANSI color codes for pretty terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_banner():
    """Print welcome banner."""
    print(f"\n{Colors.CYAN}{'='*70}")
    print(f"{Colors.BOLD}🏥 FEDERATED LEARNING HEALTHCARE SYSTEM - QUICK START{Colors.END}")
    print(f"{Colors.CYAN}{'='*70}{Colors.END}\n")

def print_section(title):
    """Print section header."""
    print(f"\n{Colors.YELLOW}{'─'*70}")
    print(f"{Colors.BOLD}{title}{Colors.END}")
    print(f"{Colors.YELLOW}{'─'*70}{Colors.END}\n")

def check_dependencies():
    """Check if required packages are installed."""
    print_section("📦 Checking Dependencies")
    required = ['flwr', 'tensorflow', 'sklearn', 'numpy', 'pandas']
    missing = []
    for package in required:
        try:
            __import__(package)
            print(f"{Colors.GREEN}✓{Colors.END} {package} installed")
        except ImportError:
            print(f"{Colors.RED}✗{Colors.END} {package} NOT installed")
            missing.append(package)
    if missing:
        print(f"\n{Colors.RED}Missing packages detected!{Colors.END}")
        print(f"To fix, please run: {Colors.CYAN}pip install {' '.join(missing)}{Colors.END}\n")
        return False
    print(f"\n{Colors.GREEN}All dependencies satisfied!{Colors.END}")
    return True

def test_models():
    """Test if models are working."""
    print_section("🧠 Testing Neural Network Models")
    try:
        from client.model import get_model
        print("Creating cancer detection model (30 features)...")
        model = get_model('cancer', (30,))
        print(f"{Colors.GREEN}✓{Colors.END} Cancer model created: {model.count_params():,} parameters")
        print("\nCreating diabetes prediction model (8 features)...")
        model = get_model('diabetes', (8,))
        print(f"{Colors.GREEN}✓{Colors.END} Diabetes model created: {model.count_params():,} parameters")
        return True
    except Exception as e:
        print(f"{Colors.RED}✗ Error: {e}{Colors.END}")
        return False

def test_data_loader():
    """Test if data loading works with the enhanced loader."""
    print_section("📊 Testing Data Loader")
    try:
        # Assuming your file is in client/ and named Enhanceddataloader.py
        from client.Enhanceddataloader import EnhancedMedicalDataLoader
        print("Testing built-in data for hospital_a (cancer)...")
        loader = EnhancedMedicalDataLoader('hospital_a', 'cancer', use_kaggle=False)
        X_train, _, _, _ = loader.load_data()
        print(f"{Colors.GREEN}✓{Colors.END} Built-in data loaded successfully. Features: {X_train.shape[1]}")
        return True
    except Exception as e:
        print(f"{Colors.RED}✗ Error: {e}{Colors.END}")
        return False

def run_demo_scenario():
    """Run a complete demo scenario, with a choice for Kaggle data."""
    print_section("🚀 Running Demo Scenario")
    print(f"{Colors.CYAN}This will demonstrate federated learning with:{Colors.END}")
    print("  • 1 FL Server")
    print("  • 2 Hospitals (hospital_a, hospital_b)")
    print("  • Disease: Diabetes")
    print("  • Rounds: 3 (for quick demo)")
    
    response = input(f"\n{Colors.YELLOW}Continue? (y/n):{Colors.END} ").lower()
    if response != 'y':
        print("Demo cancelled.")
        return

    use_kaggle_response = input(f"{Colors.YELLOW}Use real Kaggle data? (y/n):{Colors.END} ").lower()
    use_kaggle = use_kaggle_response == 'y'
    
    print(f"\n{Colors.GREEN}Starting demo using {'Kaggle' if use_kaggle else 'Built-in'} data...{Colors.END}\n")
    
    # Use sys.executable to ensure we use the python from the virtual env
    python_executable = sys.executable
    
    client_script_name = 'BasicFLClient-basic_client.py'
    server_script_name = 'FLserver.py'

    client_cmd_base = [python_executable, client_script_name, '--disease', 'diabetes']
    if use_kaggle:
        client_cmd_base.append('--use-kaggle')

    # Start server
    print(f"{Colors.BLUE}[SERVER]{Colors.END} Starting FL server...")
    server_process = subprocess.Popen(
        [python_executable, server_script_name, '--rounds', '3', '--min_clients', '2'], 
        text=True
    )
    time.sleep(4)
    
    # Start client 1
    print(f"{Colors.BLUE}[CLIENT 1]{Colors.END} Starting hospital_a...")
    client1_cmd = client_cmd_base + ['--hospital', 'hospital_a']
    client1_process = subprocess.Popen(client1_cmd, text=True)
    time.sleep(2)
    
    # Start client 2
    print(f"{Colors.BLUE}[CLIENT 2]{Colors.END} Starting hospital_b...")
    client2_cmd = client_cmd_base + ['--hospital', 'hospital_b']
    client2_process = subprocess.Popen(client2_cmd, text=True)
    
    print(f"\n{Colors.GREEN}All processes started!{Colors.END}")
    print(f"{Colors.YELLOW}Training in progress... Press Ctrl+C to stop early.{Colors.END}\n")
    
    try:
        server_process.wait()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Demo interrupted by user... shutting down processes.{Colors.END}")
    finally:
        server_process.kill()
        client1_process.kill()
        client2_process.kill()
        print(f"\n{Colors.GREEN}✓ Demo finished and processes stopped.{Colors.END}")

def show_manual_instructions():
    """Show manual testing instructions, including the Kaggle flag."""
    print_section("📖 Manual Testing Instructions")
    print(f"{Colors.CYAN}To test manually, open three separate terminals:{Colors.END}\n")
    print(f"{Colors.BOLD}Terminal 1 (Server):{Colors.END}")
    print(f"{Colors.GREEN}python FLserver.py --rounds 5 --min_clients 2{Colors.END}\n")
    print(f"{Colors.BOLD}Terminal 2 (Client 1 with Kaggle Data):{Colors.END}")
    print(f"{Colors.GREEN}python BasicFLClient-basic_client.py --hospital hospital_a --disease diabetes --use-kaggle{Colors.END}\n")
    print(f"{Colors.BOLD}Terminal 3 (Client 2 with Built-in Data):{Colors.END}")
    print(f"{Colors.GREEN}python BasicFLClient-basic_client.py --hospital hospital_b --disease diabetes{Colors.END}\n")

def main_menu():
    """Display main menu and handle user choices."""
    while True:
        print_banner()
        print(f"{Colors.BOLD}Choose an option:{Colors.END}\n")
        print("1. Check Dependencies")
        print("2. Test Models")
        print("3. Test Data Loader")
        print("4. Run Quick Demo (Automated)")
        print("5. Show Manual Testing Instructions")
        print("6. Run Full System Check")
        print("7. Exit")
        
        choice = input(f"\n{Colors.YELLOW}Enter choice (1-7):{Colors.END} ")
        
        if choice == '1':
            check_dependencies()
        elif choice == '2':
            test_models()
        elif choice == '3':
            test_data_loader()
        elif choice == '4':
            run_demo_scenario()
        elif choice == '5':
            show_manual_instructions()
        elif choice == '6':
            print_section("⚙️ Running Full System Check")
            if check_dependencies() and test_models() and test_data_loader():
                print(f"\n{Colors.GREEN}✓ All system checks passed!{Colors.END}")
            else:
                print(f"\n{Colors.RED}✗ One or more system checks failed.{Colors.END}")
        elif choice == '7':
            print(f"\n{Colors.CYAN}Exiting... Goodbye!{Colors.END}")
            break
        else:
            print(f"\n{Colors.RED}Invalid choice. Please enter a number between 1 and 7.{Colors.END}")
            time.sleep(2)
        
        if choice != '7':
            input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.END}")

if __name__ == "__main__":
    main_menu()