"""
Quick Start Script for Federated Learning System
This script helps you test the system easily without manual terminal management.
"""

import subprocess
import time
import sys
import os
from threading import Thread

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
        from client.model import get_model # Assuming a generic get_model
        
        print("Creating cancer detection model...")
        model = get_model('cancer', (30,))
        print(f"{Colors.GREEN}✓{Colors.END} Cancer model created: {model.count_params():,} parameters")
        
        print("\nCreating diabetes prediction model...")
        model = get_model('diabetes', (8,))
        print(f"{Colors.GREEN}✓{Colors.END} Diabetes model created: {model.count_params():,} parameters")
        
        return True
    except Exception as e:
        print(f"{Colors.RED}✗ Error: Could not import or create models.{Colors.END}")
        print(f"  Details: {e}")
        return False

def test_data_loader():
    """Test if data loading works."""
    print_section("📊 Testing Data Loader")
    
    try:
        from client.data_loader import MedicalDataLoader
        
        print("Loading data for hospital_a (cancer)...")
        loader = MedicalDataLoader('hospital_a', 'cancer')
        X_train, X_test, y_train, y_test = loader.load_data()
        
        print(f"{Colors.GREEN}✓{Colors.END} Data loaded successfully")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")
        print(f"  Features: {X_train.shape[1]}")
        
        return True
    except Exception as e:
        print(f"{Colors.RED}✗ Error: Could not load data.{Colors.END}")
        print(f"  Details: {e}")
        return False

def run_demo_scenario():
    """Run a complete demo scenario."""
    print_section("🚀 Running Demo Scenario")
    
    print(f"{Colors.CYAN}This will demonstrate federated learning with:{Colors.END}")
    print("  • 1 FL Server")
    print("  • 2 Hospitals (hospital_a, hospital_b)")
    print("  • Disease: Cancer Detection")
    print("  • Rounds: 3 (for quick demo)")
    
    response = input(f"\n{Colors.YELLOW}Continue? (y/n):{Colors.END} ").lower()
    if response != 'y':
        print("Demo cancelled.")
        return
    
    print(f"\n{Colors.GREEN}Starting demo...{Colors.END}\n")
    
    # Start server in background
    print(f"{Colors.BLUE}[SERVER]{Colors.END} Starting FL server...")
    server_process = subprocess.Popen(
        ['python', 'server.py', '--rounds', '3', '--min_clients', '2'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for server to start
    time.sleep(3)
    
    # Start client 1
    print(f"{Colors.BLUE}[CLIENT 1]{Colors.END} Starting hospital_a...")
    client1_process = subprocess.Popen(
        ['python', 'BasicFLClient-basic_client.py', '--hospital', 'hospital_a', '--disease', 'cancer'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    time.sleep(2)
    
    # Start client 2
    print(f"{Colors.BLUE}[CLIENT 2]{Colors.END} Starting hospital_b...")
    client2_process = subprocess.Popen(
        ['python', 'BasicFLClient-basic_client.py', '--hospital', 'hospital_b', '--disease', 'cancer'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    print(f"\n{Colors.GREEN}All processes started!{Colors.END}")
    print(f"{Colors.YELLOW}Training in progress... (this may take a few minutes){Colors.END}\n")
    
    # Wait for completion
    try:
        server_process.wait(timeout=300)  # 5 minute timeout
        client1_process.wait(timeout=10)
        client2_process.wait(timeout=10)
        
        print(f"\n{Colors.GREEN}✓ Demo completed successfully!{Colors.END}")
        
    except subprocess.TimeoutExpired:
        print(f"\n{Colors.RED}✗ Demo timed out{Colors.END}")
        server_process.kill()
        client1_process.kill()
        client2_process.kill()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Demo interrupted by user{Colors.END}")
        server_process.kill()
        client1_process.kill()
        client2_process.kill()

def show_manual_instructions():
    """Show manual testing instructions."""
    print_section("📖 Manual Testing Instructions")
    
    print(f"{Colors.CYAN}To test manually, open three separate terminals and run the following commands:{Colors.END}\n")
    
    print(f"{Colors.BOLD}Terminal 1 (Server):{Colors.END}")
    print(f"{Colors.GREEN}python server.py --rounds 5 --min_clients 2{Colors.END}\n")
    
    print(f"{Colors.BOLD}Terminal 2 (Client 1):{Colors.END}")
    print(f"{Colors.GREEN}python BasicFLClient-basic_client.py --hospital hospital_a --disease cancer{Colors.END}\n")
    
    print(f"{Colors.BOLD}Terminal 3 (Client 2):{Colors.END}")
    print(f"{Colors.GREEN}python BasicFLClient-basic_client.py --hospital hospital_b --disease cancer{Colors.END}\n")

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
            input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.END}")
        elif choice == '2':
            test_models()
            input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.END}")
        elif choice == '3':
            test_data_loader()
            input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.END}")
        elif choice == '4':
            run_demo_scenario()
            input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.END}")
        elif choice == '5':
            show_manual_instructions()
            input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.END}")
        elif choice == '6':
            print_section("⚙️ Running Full System Check")
            if check_dependencies() and test_models() and test_data_loader():
                print(f"\n{Colors.GREEN}✓ All system checks passed!{Colors.END}")
            else:
                print(f"\n{Colors.RED}✗ One or more system checks failed.{Colors.END}")
            input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.END}")
        elif choice == '7':
            print(f"\n{Colors.CYAN}Exiting... Goodbye!{Colors.END}")
            break
        else:
            print(f"\n{Colors.RED}Invalid choice. Please enter a number between 1 and 7.{Colors.END}")
            time.sleep(2)

if __name__ == "__main__":
    # Add the parent directory to the path to allow imports from client/ and server/
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    main_menu()