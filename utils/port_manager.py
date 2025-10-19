"""
Level 3: Port Manager
Manages port allocation for FL servers to avoid conflicts.

CONCEPT: Like a university assigning classroom numbers.
Each FL project needs its own port (classroom), and we need to make sure
no two projects use the same port.
"""

import socket
import threading
from typing import Optional, Set


class PortManager:
    """
    Manages port allocation for FL servers.
    Thread-safe implementation.
    """
    
    def __init__(
        self,
        start_port: int = 9000,
        end_port: int = 9100,
        reserved_ports: Optional[Set[int]] = None
    ):
        """
        Initialize port manager.
        
        Args:
            start_port: Starting port number for allocation
            end_port: Ending port number for allocation
            reserved_ports: Pre-reserved ports to avoid
        """
        self.start_port = start_port
        self.end_port = end_port
        self.allocated_ports: Set[int] = reserved_ports or set()
        self.lock = threading.RLock()
        
        print(f"✓ Port Manager initialized (range: {start_port}-{end_port})")
    
    def is_port_available(self, port: int) -> bool:
        """
        Check if a port is available for use.
        
        Args:
            port: Port number to check
            
        Returns:
            True if port is available
        """
        # Check if already allocated
        if port in self.allocated_ports:
            return False
        
        # Try to bind to the port
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(('', port))
                return True
        except OSError:
            return False
    
    def allocate_port(self) -> Optional[int]:
        """
        Allocate an available port.
        
        Returns:
            Port number if available, None otherwise
        """
        with self.lock:
            for port in range(self.start_port, self.end_port + 1):
                if self.is_port_available(port):
                    self.allocated_ports.add(port)
                    print(f"✓ Port allocated: {port}")
                    return port
            
            print(f"✗ No available ports in range {self.start_port}-{self.end_port}")
            return None
    
    def allocate_specific_port(self, port: int) -> bool:
        """
        Allocate a specific port if available.
        
        Args:
            port: Desired port number
            
        Returns:
            True if allocated successfully
        """
        with self.lock:
            if port < self.start_port or port > self.end_port:
                print(f"✗ Port {port} outside allowed range")
                return False
            
            if self.is_port_available(port):
                self.allocated_ports.add(port)
                print(f"✓ Specific port allocated: {port}")
                return True
            
            print(f"✗ Port {port} not available")
            return False
    
    def release_port(self, port: int) -> bool:
        """
        Release an allocated port.
        
        Args:
            port: Port number to release
            
        Returns:
            True if released successfully
        """
        with self.lock:
            if port in self.allocated_ports:
                self.allocated_ports.remove(port)
                print(f"✓ Port released: {port}")
                return True
            
            print(f"⚠️  Port {port} was not allocated")
            return False
    
    def get_allocated_ports(self) -> Set[int]:
        """Get set of currently allocated ports."""
        with self.lock:
            return self.allocated_ports.copy()
    
    def get_available_count(self) -> int:
        """Get number of available ports."""
        with self.lock:
            total_ports = self.end_port - self.start_port + 1
            return total_ports - len(self.allocated_ports)


# ==================== TESTING ====================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING PORT MANAGER - LEVEL 3")
    print("="*70)
    
    # Create port manager
    pm = PortManager(start_port=9000, end_port=9010)
    
    # Test 1: Allocate ports
    print("\n>>> Test 1: Allocating Ports")
    port1 = pm.allocate_port()
    port2 = pm.allocate_port()
    port3 = pm.allocate_port()
    print(f"Allocated ports: {port1}, {port2}, {port3}")
    
    # Test 2: Check availability
    print("\n>>> Test 2: Checking Port Availability")
    print(f"Port {port1} available? {pm.is_port_available(port1)}")
    print(f"Port 9099 available? {pm.is_port_available(9099)}")
    
    # Test 3: Release port
    print("\n>>> Test 3: Releasing Port")
    pm.release_port(port2)
    print(f"Port {port2} available? {pm.is_port_available(port2)}")
    
    # Test 4: Allocate specific port
    print("\n>>> Test 4: Allocating Specific Port")
    success = pm.allocate_specific_port(9005)
    print(f"Allocated port 9005: {success}")
    
    # Test 5: Statistics
    print("\n>>> Test 5: Statistics")
    print(f"Allocated ports: {pm.get_allocated_ports()}")
    print(f"Available ports: {pm.get_available_count()}")
    
    print("\n" + "="*70)
    print("✓ PORT MANAGER WORKING!")
    print("="*70 + "\n")