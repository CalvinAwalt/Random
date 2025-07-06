from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncio
import random

app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket manager
class ConnectionManager:
    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

# Simulated AI state
ai_state = {
    "intelligence": 0.92,
    "ethical_compliance": 0.98,
    "neural_complexity": 42000,
    "energy_efficiency": 0.33
}

# AI Core Logic
def accelerate_learning():
    ai_state["intelligence"] += random.uniform(0.01, 0.03)
    ai_state["neural_complexity"] += random.randint(500, 1500)
    ai_state["energy_efficiency"] -= random.uniform(0.01, 0.02)

def self_optimize():
    ai_state["energy_efficiency"] += random.uniform(0.01, 0.03)
    ai_state["neural_complexity"] -= random.randint(200, 600)

def validate_ethics():
    ai_state["ethical_compliance"] += random.uniform(0.005, 0.015)

def transform_architecture():
    ai_state["neural_complexity"] += random.randint(1000, 3000)
    ai_state["intelligence"] += random.uniform(0.02, 0.05)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            command = eval(data).get("command", "")
            
            if command == "accelerate_learning":
                accelerate_learning()
            elif command == "self_optimize":
                self_optimize()
            elif command == "validate_ethics":
                validate_ethics()
            elif command == "transform_architecture":
                transform_architecture()
            
            # Broadcast updated metrics
            await manager.broadcast(f"""{{
                "type": "metrics_update",
                "intelligence": {ai_state['intelligence']},
                "ethical": {ai_state['ethical_compliance']},
                "complexity": {ai_state['neural_complexity']},
                "energy": {ai_state['energy_efficiency']}
            }}""")
    except Exception as e:
        print(f"Error: {e}")
        manager.disconnect(websocket)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)