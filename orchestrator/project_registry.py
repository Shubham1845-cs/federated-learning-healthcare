"""
Level 3: Project Registry
The "Course Catalog" - tracks all FL projects and their status.

CONCEPT: Like a university's master schedule that tracks:
- All available courses (FL projects)
- Which classroom each course is in (port numbers)
- How many students enrolled (connected clients)
- Current status (waiting, active, completed)
"""

from typing import Dict, List, Optional
from datetime import datetime
import threading
from pathlib import Path
import json

from orchestrator.models import (
    ProjectInfo,
    ProjectStatus,
    ClientInfo,
    ProjectResponse,
    generate_project_id
)


class ProjectRegistry:
    """
    Central registry for all federated learning projects.
    Thread-safe implementation for concurrent access.
    """
    
    def __init__(self, storage_path: str = "orchestrator/data"):
        """
        Initialize project registry.
        
        Args:
            storage_path: Path to store project data
        """
        self.projects: Dict[str, ProjectInfo] = {}
        self.clients: Dict[str, ClientInfo] = {}
        self.lock = threading.RLock()
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        print("✓ Project Registry initialized")
        self._load_persisted_projects()
    
    # ==================== PROJECT CRUD OPERATIONS ====================
    
    def create_project(
        self,
        project_name: str,
        disease_type: str,
        strategy: str,
        num_rounds: int,
        min_clients: int,
        local_epochs: int = 5,
        batch_size: int = 32,
        **config_kwargs
    ) -> ProjectInfo:
        """
        Create a new FL project.
        
        Args:
            project_name: Name of the project
            disease_type: Disease to predict
            strategy: FL strategy name
            num_rounds: Number of training rounds
            min_clients: Minimum clients required
            local_epochs: Epochs per round per client
            batch_size: Training batch size
            **config_kwargs: Additional configuration
            
        Returns:
            Created ProjectInfo
        """
        with self.lock:
            # Generate unique project ID
            project_id = generate_project_id(project_name, disease_type)
            
            # Check if project already exists
            if project_id in self.projects:
                raise ValueError(f"Project {project_id} already exists")
            
            # Create project
            project = ProjectInfo(
                project_id=project_id,
                project_name=project_name,
                disease_type=disease_type,
                strategy=strategy,
                status=ProjectStatus.CREATED,
                num_rounds=num_rounds,
                min_clients=min_clients,
                local_epochs=local_epochs,
                batch_size=batch_size,
                created_at=datetime.now(),
                config=config_kwargs
            )
            
            self.projects[project_id] = project
            self._persist_project(project)
            
            print(f"✓ Project created: {project_id} ({project_name})")
            return project
    
    def get_project(self, project_id: str) -> Optional[ProjectInfo]:
        """
        Get project by ID.
        
        Args:
            project_id: Project identifier
            
        Returns:
            ProjectInfo if found, None otherwise
        """
        with self.lock:
            return self.projects.get(project_id)
    
    def get_all_projects(self) -> List[ProjectInfo]:
        """Get all projects."""
        with self.lock:
            return list(self.projects.values())
    
    def get_active_projects(self) -> List[ProjectInfo]:
        """Get projects that are currently training."""
        with self.lock:
            return [
                p for p in self.projects.values()
                if p.status in [ProjectStatus.TRAINING, ProjectStatus.WAITING_FOR_CLIENTS]
            ]
    
    def update_project_status(
        self,
        project_id: str,
        status: ProjectStatus,
        **updates
    ) -> bool:
        """
        Update project status and other fields.
        
        Args:
            project_id: Project identifier
            status: New status
            **updates: Other fields to update
            
        Returns:
            True if updated, False if project not found
        """
        with self.lock:
            project = self.projects.get(project_id)
            if not project:
                return False
            
            project.status = status
            
            # Update timestamps
            if status == ProjectStatus.TRAINING and not project.started_at:
                project.started_at = datetime.now()
            elif status == ProjectStatus.COMPLETED and not project.completed_at:
                project.completed_at = datetime.now()
            
            # Update other fields
            for key, value in updates.items():
                if hasattr(project, key):
                    setattr(project, key, value)
            
            self._persist_project(project)
            print(f"✓ Project {project_id} status updated: {status.value}")
            return True
    
    def delete_project(self, project_id: str) -> bool:
        """
        Delete a project.
        
        Args:
            project_id: Project identifier
            
        Returns:
            True if deleted, False if not found
        """
        with self.lock:
            if project_id in self.projects:
                del self.projects[project_id]
                # Remove persisted file
                project_file = self.storage_path / f"{project_id}.json"
                if project_file.exists():
                    project_file.unlink()
                print(f"✓ Project {project_id} deleted")
                return True
            return False
    
    # ==================== CLIENT MANAGEMENT ====================
    
    def register_client(
        self,
        hospital_id: str,
        project_id: str
    ) -> bool:
        """
        Register a client to a project.
        
        Args:
            hospital_id: Hospital identifier
            project_id: Project to join
            
        Returns:
            True if registered successfully
        """
        with self.lock:
            project = self.projects.get(project_id)
            if not project:
                return False
            
            # Add client to project
            if hospital_id not in project.connected_clients:
                project.connected_clients.append(hospital_id)
            
            # Create client info
            client_key = f"{hospital_id}_{project_id}"
            self.clients[client_key] = ClientInfo(
                hospital_id=hospital_id,
                project_id=project_id,
                connected_at=datetime.now(),
                last_seen=datetime.now()
            )
            
            self._persist_project(project)
            print(f"✓ Client {hospital_id} registered to project {project_id}")
            return True
    
    def unregister_client(
        self,
        hospital_id: str,
        project_id: str
    ) -> bool:
        """
        Unregister a client from a project.
        
        Args:
            hospital_id: Hospital identifier
            project_id: Project ID
            
        Returns:
            True if unregistered
        """
        with self.lock:
            project = self.projects.get(project_id)
            if not project:
                return False
            
            if hospital_id in project.connected_clients:
                project.connected_clients.remove(hospital_id)
            
            client_key = f"{hospital_id}_{project_id}"
            if client_key in self.clients:
                del self.clients[client_key]
            
            self._persist_project(project)
            return True
    
    def update_client_activity(
        self,
        hospital_id: str,
        project_id: str
    ) -> None:
        """Update client last seen timestamp."""
        with self.lock:
            client_key = f"{hospital_id}_{project_id}"
            if client_key in self.clients:
                self.clients[client_key].last_seen = datetime.now()
    
    def get_project_clients(self, project_id: str) -> List[str]:
        """Get list of clients for a project."""
        with self.lock:
            project = self.projects.get(project_id)
            return project.connected_clients if project else []
    
    # ==================== METRICS MANAGEMENT ====================
    
    def add_round_metrics(
        self,
        project_id: str,
        round_number: int,
        metrics: Dict
    ) -> bool:
        """
        Add metrics for a training round.
        
        Args:
            project_id: Project identifier
            round_number: Round number
            metrics: Metrics dictionary
            
        Returns:
            True if added successfully
        """
        with self.lock:
            project = self.projects.get(project_id)
            if not project:
                return False
            
            metrics_entry = {
                'round': round_number,
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics
            }
            
            project.metrics_history.append(metrics_entry)
            project.current_round = round_number
            
            self._persist_project(project)
            return True
    
    def get_project_metrics(self, project_id: str) -> List[Dict]:
        """Get metrics history for a project."""
        with self.lock:
            project = self.projects.get(project_id)
            return project.metrics_history if project else []
    
    # ==================== PORT MANAGEMENT ====================
    
    def assign_port(self, project_id: str, port: int) -> bool:
        """
        Assign FL server port to project.
        
        Args:
            project_id: Project identifier
            port: Port number
            
        Returns:
            True if assigned
        """
        with self.lock:
            project = self.projects.get(project_id)
            if not project:
                return False
            
            project.fl_server_port = port
            self._persist_project(project)
            return True
    
    def get_used_ports(self) -> List[int]:
        """Get list of ports currently in use."""
        with self.lock:
            return [
                p.fl_server_port
                for p in self.projects.values()
                if p.fl_server_port is not None
            ]
    
    # ==================== CONVERSION & PERSISTENCE ====================
    
    def to_project_response(self, project: ProjectInfo) -> ProjectResponse:
        """Convert ProjectInfo to API response."""
        return ProjectResponse(
            project_id=project.project_id,
            project_name=project.project_name,
            disease_type=project.disease_type,
            strategy=project.strategy,
            status=project.status.value,
            num_rounds=project.num_rounds,
            current_round=project.current_round,
            min_clients=project.min_clients,
            connected_clients=len(project.connected_clients),
            fl_server_port=project.fl_server_port,
            created_at=project.created_at.isoformat(),
            started_at=project.started_at.isoformat() if project.started_at else None,
            completed_at=project.completed_at.isoformat() if project.completed_at else None
        )
    
    def _persist_project(self, project: ProjectInfo) -> None:
        """Save project to disk."""
        project_file = self.storage_path / f"{project.project_id}.json"
        
        # Convert to dict (excluding non-serializable fields)
        project_dict = project.dict(exclude={'fl_server_process'})
        
        # Convert datetime to string
        for key in ['created_at', 'started_at', 'completed_at']:
            if project_dict.get(key):
                project_dict[key] = project_dict[key].isoformat()
        
        with open(project_file, 'w') as f:
            json.dump(project_dict, f, indent=2)
    
    def _load_persisted_projects(self) -> None:
        """Load projects from disk on startup."""
        for project_file in self.storage_path.glob("*.json"):
            try:
                with open(project_file, 'r') as f:
                    project_dict = json.load(f)
                
                # Convert string dates back to datetime
                for key in ['created_at', 'started_at', 'completed_at']:
                    if project_dict.get(key):
                        project_dict[key] = datetime.fromisoformat(project_dict[key])
                
                project = ProjectInfo(**project_dict)
                self.projects[project.project_id] = project
                print(f"✓ Loaded persisted project: {project.project_id}")
            except Exception as e:
                print(f"✗ Error loading project {project_file}: {e}")
    
    # ==================== STATISTICS ====================
    
    def get_statistics(self) -> Dict:
        """Get registry statistics."""
        with self.lock:
            return {
                'total_projects': len(self.projects),
                'active_projects': len(self.get_active_projects()),
                'total_clients': len(self.clients),
                'projects_by_status': {
                    status.value: sum(1 for p in self.projects.values() if p.status == status)
                    for status in ProjectStatus
                },
                'projects_by_disease': {
                    disease: sum(1 for p in self.projects.values() if p.disease_type == disease)
                    for disease in set(p.disease_type for p in self.projects.values())
                }
            }


# ==================== TESTING ====================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING PROJECT REGISTRY - LEVEL 3")
    print("="*70)
    
    # Create registry
    registry = ProjectRegistry(storage_path="test_data")
    
    # Test 1: Create projects
    print("\n>>> Test 1: Creating Projects")
    proj1 = registry.create_project(
        project_name="Diabetes_Study_2024",
        disease_type="diabetes",
        strategy="fedavg",
        num_rounds=10,
        min_clients=2
    )
    print(f"Created: {proj1.project_id}")
    
    proj2 = registry.create_project(
        project_name="Heart_Disease_Trial",
        disease_type="heart_disease",
        strategy="fedprox",
        num_rounds=15,
        min_clients=3
    )
    print(f"Created: {proj2.project_id}")
    
    # Test 2: Register clients
    print("\n>>> Test 2: Registering Clients")
    registry.register_client("hospital_a", proj1.project_id)
    registry.register_client("hospital_b", proj1.project_id)
    registry.register_client("clinic_1", proj2.project_id)
    
    # Test 3: Update status
    print("\n>>> Test 3: Updating Project Status")
    registry.assign_port(proj1.project_id, 9001)
    registry.update_project_status(proj1.project_id, ProjectStatus.TRAINING)
    
    # Test 4: Get projects
    print("\n>>> Test 4: Retrieving Projects")
    all_projects = registry.get_all_projects()
    print(f"Total projects: {len(all_projects)}")
    
    active = registry.get_active_projects()
    print(f"Active projects: {len(active)}")
    
    # Test 5: Statistics
    print("\n>>> Test 5: Statistics")
    stats = registry.get_statistics()
    print(json.dumps(stats, indent=2))
    
    print("\n" + "="*70)
    print("✓ PROJECT REGISTRY WORKING!")
    print("="*70 + "\n")