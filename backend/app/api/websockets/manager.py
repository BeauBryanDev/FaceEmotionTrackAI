from fastapi import WebSocket
from typing import Dict

class ConnectionManager:
    """
    Manages active WebSocket connections to ensure clean lifecycle 
    and facilitate potential broadcast operations.
    """
    def __init__(self):
        # Maps user_id to their active WebSocket connection
        self.active_connections: Dict[int, WebSocket] = {}

    async def connect(self, user_id: int, websocket: WebSocket):
        """Accepts the connection and registers the user."""
        await websocket.accept()
        self.active_connections[user_id] = websocket
        print(f"User {user_id} connected. Active connections: {len(self.active_connections)}")

    def disconnect(self, user_id: int):
        """Removes the user from the registry."""
        if user_id in self.active_connections:
            del self.active_connections[user_id]
            print(f"User {user_id} disconnected. Active connections: {len(self.active_connections)}")

    async def send_personal_json(self, data: dict, user_id: int):
        """Sends an analysis result to a specific user."""
        websocket = self.active_connections.get(user_id)
        if websocket:
            await websocket.send_json(data)

# Global instance to be used across the application
manager = ConnectionManager()