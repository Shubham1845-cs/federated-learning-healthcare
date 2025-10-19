"""
Level 3: Pydantic Models for Orchestrator API
Data models for requests, responses, and internal state management.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime


class ProjectStatus(str, Enum):
    """Status of a federated learning project."""
    CREATED = "created"
    WAITING_FOR_CLIENTS = "waiting_for_clients"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


class DiseaseType(str, Enum):
    """Supported disease types."""
    DIABETES = "diabetes"
    HEART_DISEASE = "heart_disease"
    CARDIOVASCULAR = "cardiovascular"
    CANCER = "cancer"


class StrategyType(str, Enum):
    """Supported FL strategies."""
    FEDAVG = "fedavg"
    FEDPROX = "fedprox"
    FEDAVGM = "fedavgm"
    ADAPTIVE = "adaptive"


# ==================== REQUEST MODELS ====================

class CreateProjectRequest(BaseModel):
    """Request to create a new FL project."""
    project_name: str = Field(..., description="Unique project name", min_length=3)
    disease_type: DiseaseType = Field(..., description="Disease to predict")
    strategy: StrategyType = Field(default=StrategyType.FEDAVG, description="FL strategy")
    num_rounds: int = Field(default=10, ge=1, le=100, description="Number of training rounds")
    min_clients: int = Field(default=2, ge=1, description="Minimum clients required")
    local_epochs: int = Field(default=5, ge=1, description="Epochs per client per round")
    batch_size: int = Field(default=32, ge=1, description="Training batch size")
    
    # Strategy-specific parameters
    proximal_mu: Optional[float] = Field(default=0.01, description="FedProx proximal term")
    momentum: Optional[float] = Field(default=0.9, description="FedAvgM momentum")
    
    class Config:
        schema_extra = {
            "example": {
                "project_name": "Diabetes_Prediction_2024",
                "disease_type": "diabetes",
                "strategy": "fedavg",
                "num_rounds": 10,
                "min_clients": 2,
                "local_epochs": 5,
                "batch_size": 32
            }
        }


class JoinProjectRequest(BaseModel):
    """Request from client to join a project."""
    hospital_id: str = Field(..., description="Unique hospital identifier", min_length=3)
    project_id: str = Field(..., description="Project ID to join")
    
    class Config:
        schema_extra = {
            "example": {
                "hospital_id": "hospital_a",
                "project_id": "diabetes_proj_001"
            }
        }


class UpdateProjectRequest(BaseModel):
    """Request to update project settings."""
    status: Optional[ProjectStatus] = None
    num_rounds: Optional[int] = Field(None, ge=1, le=100)
    min_clients: Optional[int] = Field(None, ge=1)


# ==================== RESPONSE MODELS ====================

class ProjectResponse(BaseModel):
    """Response containing project information."""
    project_id: str
    project_name: str
    disease_type: str
    strategy: str
    status: str
    num_rounds: int
    current_round: int
    min_clients: int
    connected_clients: int
    fl_server_port: Optional[int]
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    
    class Config:
        schema_extra = {
            "example": {
                "project_id": "diabetes_proj_001",
                "project_name": "Diabetes_Prediction_2024",
                "disease_type": "diabetes",
                "strategy": "fedavg",
                "status": "training",
                "num_rounds": 10,
                "current_round": 3,
                "min_clients": 2,
                "connected_clients": 2,
                "fl_server_port": 9001,
                "created_at": "2024-10-19T10:30:00",
                "started_at": "2024-10-19T10:32:00",
                "completed_at": None
            }
        }


class JoinProjectResponse(BaseModel):
    """Response when client joins a project."""
    success: bool
    message: str
    project_id: str
    fl_server_address: str
    fl_server_port: int
    project_config: Dict[str, Any]
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Successfully joined project",
                "project_id": "diabetes_proj_001",
                "fl_server_address": "localhost",
                "fl_server_port": 9001,
                "project_config": {
                    "disease_type": "diabetes",
                    "strategy": "fedavg",
                    "local_epochs": 5,
                    "batch_size": 32
                }
            }
        }


class ProjectListResponse(BaseModel):
    """Response containing list of projects."""
    total_projects: int
    projects: List[ProjectResponse]


class ProjectMetricsResponse(BaseModel):
    """Response containing project metrics."""
    project_id: str
    current_round: int
    metrics: Dict[str, Any]
    history: List[Dict[str, Any]]


class HealthCheckResponse(BaseModel):
    """Health check response."""
    status: str
    orchestrator_version: str
    active_projects: int
    total_clients: int
    uptime_seconds: float
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "orchestrator_version": "3.0.0",
                "active_projects": 3,
                "total_clients": 7,
                "uptime_seconds": 3600.5
            }
        }


# ==================== INTERNAL MODELS ====================

class ProjectInfo(BaseModel):
    """Internal representation of a project."""
    project_id: str
    project_name: str
    disease_type: str
    strategy: str
    status: ProjectStatus
    num_rounds: int
    current_round: int = 0
    min_clients: int
    local_epochs: int
    batch_size: int
    fl_server_port: Optional[int] = None
    fl_server_process: Optional[Any] = None
    connected_clients: List[str] = []
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metrics_history: List[Dict] = []
    config: Dict[str, Any] = {}
    
    class Config:
        arbitrary_types_allowed = True


class ClientInfo(BaseModel):
    """Information about a connected client."""
    hospital_id: str
    project_id: str
    connected_at: datetime
    last_seen: datetime
    status: str = "connected"
    rounds_completed: int = 0


# ==================== ERROR MODELS ====================

class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: str
    timestamp: str
    
    class Config:
        schema_extra = {
            "example": {
                "error": "ProjectNotFound",
                "detail": "Project with ID 'xyz' does not exist",
                "timestamp": "2024-10-19T10:30:00"
            }
        }


# ==================== UTILITY FUNCTIONS ====================

def generate_project_id(project_name: str, disease_type: str) -> str:
    """
    Generate unique project ID.
    
    Args:
        project_name: Name of the project
        disease_type: Type of disease
        
    Returns:
        Unique project ID
    """
    import hashlib
    from datetime import datetime
    
    timestamp = datetime.now().isoformat()
    raw_id = f"{project_name}_{disease_type}_{timestamp}"
    hash_id = hashlib.md5(raw_id.encode()).hexdigest()[:8]
    
    return f"{disease_type}_{hash_id}"


# ==================== TESTING ====================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING PYDANTIC MODELS - LEVEL 3")
    print("="*70)
    
    # Test CreateProjectRequest
    print("\n>>> Test 1: Create Project Request")
    create_req = CreateProjectRequest(
        project_name="Test_Diabetes_Project",
        disease_type=DiseaseType.DIABETES,
        strategy=StrategyType.FEDAVG,
        num_rounds=10,
        min_clients=2
    )
    print(create_req.json(indent=2))
    
    # Test JoinProjectRequest
    print("\n>>> Test 2: Join Project Request")
    join_req = JoinProjectRequest(
        hospital_id="hospital_a",
        project_id="diabetes_proj_001"
    )
    print(join_req.json(indent=2))
    
    # Test ProjectResponse
    print("\n>>> Test 3: Project Response")
    proj_resp = ProjectResponse(
        project_id="diabetes_proj_001",
        project_name="Test_Diabetes_Project",
        disease_type="diabetes",
        strategy="fedavg",
        status="training",
        num_rounds=10,
        current_round=3,
        min_clients=2,
        connected_clients=2,
        fl_server_port=9001,
        created_at=datetime.now().isoformat(),
        started_at=datetime.now().isoformat(),
        completed_at=None
    )
    print(proj_resp.json(indent=2))
    
    # Test project ID generation
    print("\n>>> Test 4: Project ID Generation")
    proj_id = generate_project_id("My_Diabetes_Study", "diabetes")
    print(f"Generated Project ID: {proj_id}")
    
    print("\n" + "="*70)
    print("✓ ALL PYDANTIC MODELS WORKING!")
    print("="*70 + "\n")