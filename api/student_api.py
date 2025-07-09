#!/usr/bin/env python3
"""
DeepFlyer Student API
REST API for student parameter tuning and live training monitoring
"""

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import asyncio
import json
import redis
import logging

# ROS2 imports
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# DeepFlyer imports
from rl_agent.config import DeepFlyerConfig

app = FastAPI(title="DeepFlyer Student API", version="1.0.0")

# Enable CORS for web UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis for real-time data
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Global config
config = DeepFlyerConfig()


class RewardParameters(BaseModel):
    """Student-tunable reward parameters"""
    hoop_approach_reward: float = config.REWARD_CONFIG['hoop_approach_reward']
    hoop_passage_reward: float = config.REWARD_CONFIG['hoop_passage_reward']
    hoop_center_bonus: float = config.REWARD_CONFIG['hoop_center_bonus']
    visual_alignment_reward: float = config.REWARD_CONFIG['visual_alignment_reward']
    forward_progress_reward: float = config.REWARD_CONFIG['forward_progress_reward']
    speed_efficiency_bonus: float = config.REWARD_CONFIG['speed_efficiency_bonus']
    lap_completion_bonus: float = config.REWARD_CONFIG['lap_completion_bonus']
    course_completion_bonus: float = config.REWARD_CONFIG['course_completion_bonus']
    
    # Penalties
    wrong_direction_penalty: float = config.REWARD_CONFIG['wrong_direction_penalty']
    hoop_miss_penalty: float = config.REWARD_CONFIG['hoop_miss_penalty']
    collision_penalty: float = config.REWARD_CONFIG['collision_penalty']
    slow_progress_penalty: float = config.REWARD_CONFIG['slow_progress_penalty']
    erratic_flight_penalty: float = config.REWARD_CONFIG['erratic_flight_penalty']


class TrainingParameters(BaseModel):
    """Training parameters"""
    learning_rate: float = 3e-4
    batch_size: int = 64
    gamma: float = 0.99
    entropy_coef: float = 0.01
    clip_epsilon: float = 0.2
    training_minutes: int = 60


class TrainingStatus(BaseModel):
    """Current training status"""
    episode: int
    total_steps: int
    episode_reward: float
    average_reward: float
    success_rate: float
    training_time_remaining: int
    is_training: bool


@app.get("/")
async def root():
    """API health check"""
    return {"status": "DeepFlyer Student API is running"}


@app.get("/config/rewards", response_model=RewardParameters)
async def get_reward_config():
    """Get current reward parameters"""
    # Get from Redis if available, otherwise use default
    stored = redis_client.get("reward_config")
    if stored:
        return json.loads(stored)
    return RewardParameters()


@app.post("/config/rewards")
async def update_reward_config(params: RewardParameters):
    """Update reward parameters (student tuning)"""
    # Validate ranges
    for field, value in params.dict().items():
        if "reward" in field or "bonus" in field:
            if not 0 <= value <= 1000:
                raise HTTPException(400, f"{field} must be between 0 and 1000")
        elif "penalty" in field:
            if not -1000 <= value <= 0:
                raise HTTPException(400, f"{field} must be between -1000 and 0")
    
    # Store in Redis
    redis_client.set("reward_config", json.dumps(params.dict()))
    
    # Publish to ROS2 for live update
    publish_reward_update(params.dict())
    
    return {"status": "Reward parameters updated successfully"}


@app.post("/training/start")
async def start_training(params: TrainingParameters):
    """Start a new training session"""
    # Validate parameters
    if params.training_minutes < 1 or params.training_minutes > 480:
        raise HTTPException(400, "Training time must be between 1 and 480 minutes")
    
    # Store training config
    redis_client.set("training_config", json.dumps(params.dict()))
    redis_client.set("training_active", "true")
    
    # Publish start command
    publish_training_command("start", params.dict())
    
    return {"status": "Training started", "estimated_completion": params.training_minutes}


@app.post("/training/stop")
async def stop_training():
    """Stop current training session"""
    redis_client.set("training_active", "false")
    publish_training_command("stop", {})
    
    return {"status": "Training stopped"}


@app.get("/training/status", response_model=TrainingStatus)
async def get_training_status():
    """Get current training status"""
    # Get from Redis
    status_data = redis_client.hgetall("training_status")
    
    if not status_data:
        return TrainingStatus(
            episode=0,
            total_steps=0,
            episode_reward=0.0,
            average_reward=0.0,
            success_rate=0.0,
            training_time_remaining=0,
            is_training=False
        )
    
    return TrainingStatus(
        episode=int(status_data.get("episode", 0)),
        total_steps=int(status_data.get("total_steps", 0)),
        episode_reward=float(status_data.get("episode_reward", 0.0)),
        average_reward=float(status_data.get("average_reward", 0.0)),
        success_rate=float(status_data.get("success_rate", 0.0)),
        training_time_remaining=int(status_data.get("time_remaining", 0)),
        is_training=status_data.get("is_training", "false") == "true"
    )


@app.get("/metrics/rewards")
async def get_reward_metrics():
    """Get detailed reward component breakdown"""
    # Get last 100 reward components
    components = []
    for i in range(100):
        data = redis_client.get(f"reward_component_{i}")
        if data:
            components.append(json.loads(data))
    
    return {"reward_components": components}


@app.websocket("/ws/live")
async def websocket_live_data(websocket: WebSocket):
    """WebSocket for live training data streaming"""
    await websocket.accept()
    
    try:
        while True:
            # Get latest data from Redis
            status = await get_training_status()
            rewards = redis_client.get("latest_reward_breakdown")
            
            data = {
                "type": "update",
                "status": status.dict(),
                "rewards": json.loads(rewards) if rewards else {}
            }
            
            await websocket.send_json(data)
            await asyncio.sleep(0.5)  # 2Hz update rate
            
    except Exception as e:
        logging.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()


# ROS2 Publishers
ros_node = None
reward_pub = None
training_pub = None


def init_ros2():
    """Initialize ROS2 node for publishing updates"""
    global ros_node, reward_pub, training_pub
    
    rclpy.init()
    ros_node = Node("student_api_node")
    
    # Publishers
    reward_pub = ros_node.create_publisher(String, "/deepflyer/reward_config_update", 10)
    training_pub = ros_node.create_publisher(String, "/deepflyer/training_command", 10)


def publish_reward_update(config: dict):
    """Publish reward configuration update to ROS2"""
    if reward_pub:
        msg = String()
        msg.data = json.dumps(config)
        reward_pub.publish(msg)


def publish_training_command(command: str, params: dict):
    """Publish training command to ROS2"""
    if training_pub:
        msg = String()
        msg.data = json.dumps({"command": command, "params": params})
        training_pub.publish(msg)


@app.on_event("startup")
async def startup_event():
    """Initialize ROS2 on startup"""
    init_ros2()
    logging.info("DeepFlyer Student API started")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if ros_node:
        ros_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 