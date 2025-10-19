"""
Level 3: Smart Client with Orchestrator Integration

This client:
1. Contacts the orchestrator API first
2. Gets assigned to the correct FL server
3. Then connects to that FL server for training

It's much smarter than Level 2 clients!
"""

import requests
import flwr as fl
import sys
import os
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client.advanced_client import AdvancedMedicalClient


class OrchestratorClient:
    """
    Smart client that uses orchestrator for project discovery.
    
    Workflow:
    1. Client asks orchestrator: "I want to join project X"
    2. Orchestrator responds: "Go to FL server at port Y"
    3. Client connects to FL server at port Y
    4. Training begins
    """
    
    def __init__(
        self,
        hospital_id: str,
        orchestrator_url: str = "http://localhost:5000"
    ):
        """
        Initialize orchestrator client.
        
        Args:
            hospital_id: Hospital identifier
            orchestrator_url: URL of orchestrator API
        """
        self.hospital_id = hospital_id
        self.orchestrator_url = orchestrator_url.rstrip('/')
        self.api_base = f"{self.orchestrator_url}/api/v1"
        
        print(f"\n{'='*70}")
        print(f"🏥 Smart FL Client (with Orchestrator)")
        print(f"Hospital: {hospital_id}")
        print(f"Orchestrator: {orchestrator_url}")
        print(f"{'='*70}\n")
    
    def check_orchestrator_health(self) -> bool:
        """
        Check if orchestrator is running.
        
        Returns:
            True if orchestrator is healthy
        """
        try:
            response = requests.get(f"{self.api_base}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"✓ Orchestrator is healthy")
                print(f"  Active projects: {data['active_projects']}")
                print(f"  Total clients: {data['total_clients']}")
                return True
            return False
        except requests.exceptions.RequestException as e:
            print(f"✗ Cannot reach orchestrator: {e}")
            return False
    
    def list_projects(self) -> list:
        """
        Get list of available projects from orchestrator.
        
        Returns:
            List of project dictionaries
        """
        try:
            response = requests.get(f"{self.api_base}/projects")
            response.raise_for_status()
            
            data = response.json()
            projects = data['projects']
            
            print(f"\n📋 Available Projects ({data['total_projects']}):")
            print(f"{'='*70}")
            
            for proj in projects:
                print(f"\nProject: {proj['project_name']}")
                print(f"  ID: {proj['project_id']}")
                print(f"  Disease: {proj['disease_type']}")
                print(f"  Strategy: {proj['strategy']}")
                print(f"  Status: {proj['status']}")
                print(f"  Rounds: {proj['current_round']}/{proj['num_rounds']}")
                print(f"  Clients: {proj['connected_clients']}/{proj['min_clients']}")
            
            return projects
        
        except requests.exceptions.RequestException as e:
            print(f"✗ Error listing projects: {e}")
            return []
    
    def join_project(self, project_id: str) -> Optional[dict]:
        """
        Join a federated learning project.
        
        Args:
            project_id: Project to join
            
        Returns:
            Project configuration if successful, None otherwise
        """
        try:
            # Send join request to orchestrator
            payload = {
                "hospital_id": self.hospital_id,
                "project_id": project_id
            }
            
            print(f"\n🔗 Requesting to join project: {project_id}")
            
            response = requests.post(
                f"{self.api_base}/projects/{project_id}/join",
                json=payload,
                timeout=10
            )
            
            response.raise_for_status()
            data = response.json()
            
            if data['success']:
                print(f"✓ Successfully joined project!")
                print(f"  Project: {project_id}")
                print(f"  FL Server: {data['fl_server_address']}:{data['fl_server_port']}")
                print(f"  Disease: {data['project_config']['disease_type']}")
                print(f"  Strategy: {data['project_config']['strategy']}")
                return data
            else:
                print(f"✗ Failed to join project: {data.get('message')}")
                return None
        
        except requests.exceptions.RequestException as e:
            print(f"✗ Error joining project: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"  Details: {e.response.text}")
            return None
    
    def start_training(
        self,
        project_id: str,
        local_epochs: Optional[int] = None,
        verbose: bool = True
    ):
        """
        Join project and start federated learning training.
        
        Args:
            project_id: Project to join
            local_epochs: Override local epochs (optional)
            verbose: Print detailed information
        """
        # Check orchestrator health
        if not self.check_orchestrator_health():
            print("\n❌ Cannot proceed - orchestrator not available")
            return
        
        # Join project
        join_response = self.join_project(project_id)
        if not join_response:
            print("\n❌ Cannot proceed - failed to join project")
            return
        
        # Extract configuration
        config = join_response['project_config']
        fl_server_address = join_response['fl_server_address']
        fl_server_port = join_response['fl_server_port']
        
        # Override local epochs if specified
        if local_epochs:
            config['local_epochs'] = local_epochs
        
        # Create FL client
        print(f"\n{'='*70}")
        print(f"Starting Federated Learning Training")
        print(f"{'='*70}")
        
        use_fedprox = config['strategy'] == 'fedprox'
        
        fl_client = AdvancedMedicalClient(
            hospital_id=self.hospital_id,
            disease_type=config['disease_type'],
            local_epochs=config.get('local_epochs', 5),
            batch_size=config.get('batch_size', 32),
            proximal_mu=config.get('proximal_mu', 0.01) if use_fedprox else 0.0,
            verbose=verbose
        )
        
        # Connect to FL server
        server_address = f"{fl_server_address}:{fl_server_port}"
        print(f"\n[{self.hospital_id}] Connecting to FL server at {server_address}...")
        
        try:
            fl.client.start_client(
                server_address=server_address,
                client=fl_client.to_client()
            )
            
            print(f"\n{'='*70}")
            print(f"✓ Training Completed Successfully!")
            print(f"{'='*70}\n")
            
            # Get final report
            fl_client.get_detailed_report()
        
        except Exception as e:
            print(f"\n❌ Error during training: {e}")
            import traceback
            traceback.print_exc()


def interactive_mode(orchestrator_url: str = "http://localhost:5000"):
    """
    Interactive mode for client.
    
    Args:
        orchestrator_url: Orchestrator API URL
    """
    print("\n" + "#"*70)
    print("# INTERACTIVE FL CLIENT - LEVEL 3")
    print("#"*70)
    
    # Get hospital ID
    hospital_id = input("\nEnter your hospital ID (e.g., hospital_a): ").strip()
    
    # Create client
    client = OrchestratorClient(hospital_id, orchestrator_url)
    
    # Check health
    if not client.check_orchestrator_health():
        print("\n❌ Cannot connect to orchestrator. Make sure it's running!")
        return
    
    # List projects
    projects = client.list_projects()
    
    if not projects:
        print("\n⚠️  No projects available. Create one first!")
        return
    
    # Choose project
    print("\n" + "="*70)
    project_id = input("Enter project ID to join: ").strip()
    
    # Start training
    client.start_training(project_id)


# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Smart FL Client with Orchestrator Integration - Level 3"
    )
    parser.add_argument(
        "--hospital",
        type=str,
        help="Hospital ID (e.g., hospital_a)"
    )
    parser.add_argument(
        "--project",
        type=str,
        help="Project ID to join"
    )
    parser.add_argument(
        "--orchestrator",
        type=str,
        default="http://localhost:5000",
        help="Orchestrator URL"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Local training epochs (override project default)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--list-projects",
        action="store_true",
        help="List available projects and exit"
    )
    
    args = parser.parse_args()
    
    # Interactive mode
    if args.interactive:
        interactive_mode(args.orchestrator)
        sys.exit(0)
    
    # List projects mode
    if args.list_projects:
        client = OrchestratorClient("temp", args.orchestrator)
        client.list_projects()
        sys.exit(0)
    
    # Normal mode - require hospital and project
    if not args.hospital or not args.project:
        print("Error: --hospital and --project are required (or use --interactive)")
        parser.print_help()
        sys.exit(1)
    
    # Create client and start training
    client = OrchestratorClient(args.hospital, args.orchestrator)
    client.start_training(args.project, local_epochs=args.epochs)