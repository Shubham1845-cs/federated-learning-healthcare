"""
Level 3: Main Orchestrator Application
The central FastAPI application that manages everything.

Run with:
    uvicorn orchestrator.main:app --host 0.0.0.0 --port 5000 --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from datetime import datetime
import signal
import sys

from orchestrator.project_registry import ProjectRegistry
from orchestrator.session_manager import FLSessionManager
from orchestrator import routes
from utils.port_manager import PortManager


# Global instances
registry: ProjectRegistry = None
session_manager: FLSessionManager = None
port_manager: PortManager = None
start_time: datetime = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown.
    """
    # Startup
    global registry, session_manager, port_manager, start_time
    
    print("\n" + "="*70)
    print("🚀 ORCHESTRATOR STARTING UP")
    print("="*70)
    
    # Initialize components
    start_time = datetime.now()
    
    print("\n1. Initializing Project Registry...")
    registry = ProjectRegistry(storage_path="orchestrator/data")
    
    print("\n2. Initializing Port Manager...")
    port_manager = PortManager(start_port=9000, end_port=9100)
    
    print("\n3. Initializing Session Manager...")
    session_manager = FLSessionManager(
        project_registry=registry,
        port_manager=port_manager,
        server_script="server/advanced_server.py"
    )
    
    print("\n4. Setting up API routes...")
    routes.set_dependencies(registry, session_manager, start_time)
    
    print("\n" + "="*70)
    print("✓ ORCHESTRATOR READY!")
    print("="*70)
    print(f"\nAPI Documentation: http://localhost:5000/docs")
    print(f"Health Check: http://localhost:5000/health")
    print("\n" + "="*70 + "\n")
    
    yield
    
    # Shutdown
    print("\n" + "="*70)
    print("🛑 ORCHESTRATOR SHUTTING DOWN")
    print("="*70)
    
    print("\nStopping all FL sessions...")
    if session_manager:
        session_manager.cleanup()
    
    print("\n✓ Shutdown complete")
    print("="*70 + "\n")


# Create FastAPI app
app = FastAPI(
    title="Federated Learning Orchestrator",
    description="Central orchestrator for managing multiple federated learning projects",
    version="3.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(routes.router, prefix="/api/v1")


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with basic information."""
    return {
        "message": "Federated Learning Orchestrator API",
        "version": "3.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
        "description": "Central orchestrator for managing multiple FL projects"
    }


# Signal handlers for graceful shutdown
def signal_handler(sig, frame):
    """Handle shutdown signals."""
    print("\n\nReceived interrupt signal. Shutting down gracefully...")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "#"*70)
    print("# FEDERATED LEARNING ORCHESTRATOR - LEVEL 3")
    print("# Starting API server...")
    print("#"*70 + "\n")
    
    uvicorn.run(
        "orchestrator.main:app",
        host="0.0.0.0",
        port=5000,
        reload=True,
        log_level="info"
    )