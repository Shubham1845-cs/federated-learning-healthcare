"""
Level 3: API Routes for Orchestrator
RESTful API endpoints for managing FL projects.

API Endpoints:
- POST   /projects                 - Create new project
- GET    /projects                 - List all projects
- GET    /projects/{id}            - Get project details
- POST   /projects/{id}/join       - Client joins project
- DELETE /projects/{id}            - Delete project
- GET    /projects/{id}/metrics    - Get project metrics
- GET    /health                   - Health check
"""

from fastapi import APIRouter, HTTPException, status
from typing import List
from datetime import datetime

from orchestrator.models import (
    CreateProjectRequest,
    JoinProjectRequest,
    ProjectResponse,
    JoinProjectResponse,
    ProjectListResponse,
    ProjectMetricsResponse,
    HealthCheckResponse,
    ErrorResponse,
    ProjectStatus
)
from orchestrator.project_registry import ProjectRegistry
from orchestrator.session_manager import FLSessionManager


# Create router
router = APIRouter()

# These will be injected by main.py
registry: ProjectRegistry = None
session_manager: FLSessionManager = None
start_time: datetime = None


def set_dependencies(reg: ProjectRegistry, sm: FLSessionManager, st: datetime):
    """Set dependencies for routes."""
    global registry, session_manager, start_time
    registry = reg
    session_manager = sm
    start_time = st


# ==================== PROJECT ENDPOINTS ====================

@router.post(
    "/projects",
    response_model=ProjectResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create New FL Project",
    tags=["Projects"]
)
async def create_project(request: CreateProjectRequest):
    """
    Create a new federated learning project.
    
    This creates a project but does NOT start the FL server yet.
    The server starts when the first client joins.
    """
    try:
        # Create project in registry
        project = registry.create_project(
            project_name=request.project_name,
            disease_type=request.disease_type.value,
            strategy=request.strategy.value,
            num_rounds=request.num_rounds,
            min_clients=request.min_clients,
            local_epochs=request.local_epochs,
            batch_size=request.batch_size,
            proximal_mu=request.proximal_mu,
            momentum=request.momentum
        )
        
        return registry.to_project_response(project)
    
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating project: {str(e)}"
        )


@router.get(
    "/projects",
    response_model=ProjectListResponse,
    summary="List All Projects",
    tags=["Projects"]
)
async def list_projects(
    status_filter: str = None,
    disease_filter: str = None
):
    """
    Get list of all FL projects.
    
    Optional filters:
    - status: Filter by project status
    - disease: Filter by disease type
    """
    try:
        projects = registry.get_all_projects()
        
        # Apply filters
        if status_filter:
            projects = [p for p in projects if p.status.value == status_filter]
        
        if disease_filter:
            projects = [p for p in projects if p.disease_type == disease_filter]
        
        # Convert to responses
        project_responses = [
            registry.to_project_response(p) for p in projects
        ]
        
        return ProjectListResponse(
            total_projects=len(project_responses),
            projects=project_responses
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing projects: {str(e)}"
        )


@router.get(
    "/projects/{project_id}",
    response_model=ProjectResponse,
    summary="Get Project Details",
    tags=["Projects"]
)
async def get_project(project_id: str):
    """Get detailed information about a specific project."""
    project = registry.get_project(project_id)
    
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found"
        )
    
    return registry.to_project_response(project)


@router.delete(
    "/projects/{project_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete Project",
    tags=["Projects"]
)
async def delete_project(project_id: str):
    """
    Delete a project.
    
    This will stop the FL server if it's running.
    """
    try:
        # Stop session if running
        if session_manager.get_session_status(project_id):
            session_manager.stop_session(project_id, graceful=False)
        
        # Delete from registry
        success = registry.delete_project(project_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project {project_id} not found"
            )
        
        return None
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting project: {str(e)}"
        )


# ==================== CLIENT ENDPOINTS ====================

@router.post(
    "/projects/{project_id}/join",
    response_model=JoinProjectResponse,
    summary="Client Joins Project",
    tags=["Clients"]
)
async def join_project(project_id: str, request: JoinProjectRequest):
    """
    Client joins a federated learning project.
    
    This is the KEY endpoint:
    1. Client sends hospital_id and project_id
    2. Server checks if project exists
    3. If FL server not running, starts it
    4. Returns FL server address and port to client
    5. Client then connects directly to FL server
    """
    try:
        # Validate project exists
        project = registry.get_project(project_id)
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project {project_id} not found"
            )
        
        # Check if project is accepting clients
        if project.status == ProjectStatus.COMPLETED:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Project has already completed"
            )
        
        if project.status == ProjectStatus.FAILED:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Project has failed"
            )
        
        # Register client
        registry.register_client(request.hospital_id, project_id)
        
        # Start FL server if not already running
        if not session_manager.get_session_status(project_id):
            success = session_manager.start_session(project_id)
            if not success:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to start FL server"
                )
        
        # Get updated project info
        project = registry.get_project(project_id)
        
        # Prepare configuration for client
        project_config = {
            "disease_type": project.disease_type,
            "strategy": project.strategy,
            "local_epochs": project.local_epochs,
            "batch_size": project.batch_size,
            "num_rounds": project.num_rounds
        }
        
        # Add strategy-specific config
        if project.strategy == "fedprox":
            project_config["proximal_mu"] = project.config.get("proximal_mu", 0.01)
        elif project.strategy == "fedavgm":
            project_config["momentum"] = project.config.get("momentum", 0.9)
        
        return JoinProjectResponse(
            success=True,
            message=f"Successfully joined project {project.project_name}",
            project_id=project_id,
            fl_server_address="localhost",
            fl_server_port=project.fl_server_port,
            project_config=project_config
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error joining project: {str(e)}"
        )


# ==================== METRICS ENDPOINTS ====================

@router.get(
    "/projects/{project_id}/metrics",
    response_model=ProjectMetricsResponse,
    summary="Get Project Metrics",
    tags=["Metrics"]
)
async def get_project_metrics(project_id: str):
    """Get training metrics for a project."""
    project = registry.get_project(project_id)
    
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found"
        )
    
    metrics_history = registry.get_project_metrics(project_id)
    
    # Get latest metrics
    latest_metrics = metrics_history[-1]['metrics'] if metrics_history else {}
    
    return ProjectMetricsResponse(
        project_id=project_id,
        current_round=project.current_round,
        metrics=latest_metrics,
        history=metrics_history
    )


# ==================== SYSTEM ENDPOINTS ====================

@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Health Check",
    tags=["System"]
)
async def health_check():
    """Check orchestrator health and get system statistics."""
    stats = registry.get_statistics()
    active_sessions = session_manager.get_active_sessions()
    
    uptime = (datetime.now() - start_time).total_seconds()
    
    return HealthCheckResponse(
        status="healthy",
        orchestrator_version="3.0.0",
        active_projects=len(active_sessions),
        total_clients=stats['total_clients'],
        uptime_seconds=uptime
    )


@router.get(
    "/statistics",
    summary="Get System Statistics",
    tags=["System"]
)
async def get_statistics():
    """Get detailed system statistics."""
    registry_stats = registry.get_statistics()
    active_sessions = session_manager.get_active_sessions()
    
    return {
        "registry": registry_stats,
        "active_sessions": active_sessions,
        "uptime_seconds": (datetime.now() - start_time).total_seconds()
    }


# ==================== ADMIN ENDPOINTS ====================

@router.post(
    "/projects/{project_id}/start",
    summary="Manually Start Project",
    tags=["Admin"]
)
async def start_project(project_id: str):
    """Manually start FL server for a project."""
    project = registry.get_project(project_id)
    
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found"
        )
    
    if session_manager.get_session_status(project_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Project session already running"
        )
    
    success = session_manager.start_session(project_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start session"
        )
    
    return {"message": "Session started successfully", "project_id": project_id}


@router.post(
    "/projects/{project_id}/stop",
    summary="Manually Stop Project",
    tags=["Admin"]
)
async def stop_project(project_id: str, graceful: bool = True):
    """Manually stop FL server for a project."""
    if not session_manager.get_session_status(project_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No active session for this project"
        )
    
    success = session_manager.stop_session(project_id, graceful=graceful)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to stop session"
        )
    
    return {"message": "Session stopped successfully", "project_id": project_id}


# ==================== TESTING ====================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("API ROUTES - LEVEL 3")
    print("="*70)
    print("\nAPI Endpoints defined:")
    print("\nProjects:")
    print("  POST   /projects                 - Create new project")
    print("  GET    /projects                 - List all projects")
    print("  GET    /projects/{id}            - Get project details")
    print("  DELETE /projects/{id}            - Delete project")
    print("\nClients:")
    print("  POST   /projects/{id}/join       - Client joins project")
    print("\nMetrics:")
    print("  GET    /projects/{id}/metrics    - Get project metrics")
    print("\nSystem:")
    print("  GET    /health                   - Health check")
    print("  GET    /statistics               - System statistics")
    print("\nAdmin:")
    print("  POST   /projects/{id}/start      - Start FL server")
    print("  POST   /projects/{id}/stop       - Stop FL server")
    print("\n" + "="*70)
    print("✓ ROUTES READY!")
    print("="*70 + "\n")