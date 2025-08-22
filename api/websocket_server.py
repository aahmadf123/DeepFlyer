#!/usr/bin/env python3
"""
WebSocket Server for Real-time ML Interface Communication
Enables Jay's frontend to receive live training updates and send parameter changes
"""

import asyncio
import json
import logging
import time
import weakref
from typing import Dict, Any, Set, Optional, Callable
from dataclasses import asdict
import threading

try:
    import websockets
    from websockets.server import WebSocketServerProtocol
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    WebSocketServerProtocol = object

from .ros_bridge import get_ros_bridge, RealTimeData
from .ml_interface import DeepFlyerMLInterface

logger = logging.getLogger(__name__)


class WebSocketManager:
    """
    WebSocket server manager for real-time ML interface communication
    
    Provides:
    - Real-time training metrics streaming
    - Parameter update endpoints 
    - Multiple client support
    - Automatic reconnection handling
    """
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.server = None
        self.running = False
        
        # Connected clients
        self.clients: Set[WebSocketServerProtocol] = set()
        
        # ML interface integration
        self.ml_interface = DeepFlyerMLInterface()
        self.ros_bridge = get_ros_bridge()
        
        # Data streaming
        self.last_metrics_update = 0.0
        self.metrics_interval = 2.0  # Send metrics every 2 seconds
        
        # Message handlers
        self.message_handlers = {
            'start_training': self._handle_start_training,
            'stop_training': self._handle_stop_training,
            'update_reward_config': self._handle_update_reward_config,
            'update_hyperparameters': self._handle_update_hyperparameters,
            'get_status': self._handle_get_status,
            'ping': self._handle_ping
        }
        
        # Streaming task
        self._streaming_task = None
        
    async def start_server(self) -> bool:
        """Start WebSocket server"""
        if not WEBSOCKETS_AVAILABLE:
            logger.error("WebSockets library not available. Install with: pip install websockets")
            return False
        
        try:
            # Setup ROS bridge callback for real-time data
            self.ros_bridge.add_data_callback(self._on_ros_data_update)
            
            # Start WebSocket server
            self.server = await websockets.serve(
                self._handle_client,
                self.host,
                self.port,
                ping_interval=20,
                ping_timeout=10
            )
            
            self.running = True
            
            # Start metrics streaming task
            self._streaming_task = asyncio.create_task(self._metrics_streaming_loop())
            
            logger.info(f"WebSocket server started on ws://{self.host}:{self.port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {e}")
            return False
    
    async def stop_server(self):
        """Stop WebSocket server"""
        self.running = False
        
        # Cancel streaming task
        if self._streaming_task:
            self._streaming_task.cancel()
            try:
                await self._streaming_task
            except asyncio.CancelledError:
                pass
        
        # Close all client connections
        if self.clients:
            await asyncio.gather(
                *[client.close() for client in self.clients],
                return_exceptions=True
            )
            self.clients.clear()
        
        # Stop server
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        
        logger.info("WebSocket server stopped")
    
    async def _handle_client(self, websocket: WebSocketServerProtocol, path: str):
        """Handle new client connection"""
        client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        logger.info(f"Client connected: {client_id}")
        
        # Add to clients set
        self.clients.add(websocket)
        
        try:
            # Send initial status
            await self._send_to_client(websocket, {
                'type': 'connection_established',
                'message': 'Connected to DeepFlyer ML Interface',
                'timestamp': time.time()
            })
            
            # Send current system status
            status = self.ml_interface.get_system_status()
            await self._send_to_client(websocket, {
                'type': 'system_status',
                'data': status,
                'timestamp': time.time()
            })
            
            # Listen for messages
            async for message in websocket:
                await self._handle_message(websocket, message)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {client_id}")
        except Exception as e:
            logger.error(f"Error handling client {client_id}: {e}")
        finally:
            # Remove from clients set
            self.clients.discard(websocket)
    
    async def _handle_message(self, websocket: WebSocketServerProtocol, message: str):
        """Handle incoming message from client"""
        try:
            data = json.loads(message)
            message_type = data.get('type')
            
            if message_type in self.message_handlers:
                response = await self.message_handlers[message_type](data)
                if response:
                    await self._send_to_client(websocket, response)
            else:
                await self._send_to_client(websocket, {
                    'type': 'error',
                    'message': f'Unknown message type: {message_type}',
                    'timestamp': time.time()
                })
                
        except json.JSONDecodeError:
            await self._send_to_client(websocket, {
                'type': 'error',
                'message': 'Invalid JSON format',
                'timestamp': time.time()
            })
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            await self._send_to_client(websocket, {
                'type': 'error',
                'message': str(e),
                'timestamp': time.time()
            })
    
    async def _send_to_client(self, websocket: WebSocketServerProtocol, data: Dict[str, Any]):
        """Send data to specific client"""
        try:
            message = json.dumps(data, default=str)
            await websocket.send(message)
        except Exception as e:
            logger.error(f"Error sending to client: {e}")
    
    async def _broadcast_to_all(self, data: Dict[str, Any]):
        """Broadcast data to all connected clients"""
        if not self.clients:
            return
        
        message = json.dumps(data, default=str)
        
        # Send to all clients concurrently
        tasks = []
        for client in self.clients.copy():  # Copy to avoid modification during iteration
            try:
                tasks.append(client.send(message))
            except Exception:
                # Remove dead clients
                self.clients.discard(client)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _metrics_streaming_loop(self):
        """Background task for streaming metrics"""
        while self.running:
            try:
                current_time = time.time()
                
                # Check if it's time to send metrics update
                if current_time - self.last_metrics_update >= self.metrics_interval:
                    await self._send_metrics_update()
                    self.last_metrics_update = current_time
                
                # Sleep for a short interval
                await asyncio.sleep(0.5)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics streaming: {e}")
                await asyncio.sleep(1.0)
    
    async def _send_metrics_update(self):
        """Send training metrics update to all clients"""
        try:
            # Get live metrics from ML interface
            metrics = self.ml_interface.get_live_training_metrics()
            progress = self.ml_interface.get_training_progress()
            reward_breakdown = self.ml_interface.get_reward_breakdown()
            
            # Prepare data for broadcast
            update_data = {
                'type': 'training_metrics',
                'data': {
                    'metrics': asdict(metrics),
                    'progress': progress,
                    'reward_breakdown': reward_breakdown,
                    'system_status': self.ml_interface.get_system_status()
                },
                'timestamp': time.time()
            }
            
            await self._broadcast_to_all(update_data)
            
        except Exception as e:
            logger.error(f"Error sending metrics update: {e}")
    
    def _on_ros_data_update(self, ros_data: RealTimeData):
        """Handle real-time data from ROS bridge"""
        # Convert ROS data to dict and broadcast immediately
        if self.clients:
            update_data = {
                'type': 'ros_data_update',
                'data': asdict(ros_data),
                'timestamp': time.time()
            }
            
            # Schedule broadcast (since this is called from ROS thread)
            asyncio.create_task(self._broadcast_to_all(update_data))
    
    # Message Handlers
    
    async def _handle_start_training(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle start training request"""
        try:
            training_minutes = data.get('training_minutes')
            reward_config_data = data.get('reward_config', {})
            hyperparameters = data.get('hyperparameters', {})
            
            # Validate training time
            if not training_minutes:
                return {
                    'type': 'error',
                    'message': 'training_minutes is required',
                    'timestamp': time.time()
                }
            
            # Create reward config if provided
            from .ml_interface import RewardConfig
            reward_config = None
            if reward_config_data:
                reward_config = RewardConfig.from_dict(reward_config_data)
            
            # Start training
            success = self.ml_interface.start_training(
                training_minutes=training_minutes,
                reward_config=reward_config,
                hyperparameters=hyperparameters
            )
            
            if success:
                # Broadcast training started to all clients
                await self._broadcast_to_all({
                    'type': 'training_started',
                    'data': {
                        'training_minutes': training_minutes,
                        'reward_config': reward_config_data,
                        'hyperparameters': hyperparameters
                    },
                    'timestamp': time.time()
                })
                
                return {
                    'type': 'training_started',
                    'message': f'Training started for {training_minutes} minutes',
                    'timestamp': time.time()
                }
            else:
                return {
                    'type': 'error',
                    'message': 'Failed to start training',
                    'timestamp': time.time()
                }
                
        except Exception as e:
            return {
                'type': 'error',
                'message': str(e),
                'timestamp': time.time()
            }
    
    async def _handle_stop_training(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle stop training request"""
        try:
            success = self.ml_interface.stop_training()
            
            if success:
                # Broadcast training stopped to all clients
                await self._broadcast_to_all({
                    'type': 'training_stopped',
                    'timestamp': time.time()
                })
                
                return {
                    'type': 'training_stopped',
                    'message': 'Training stopped successfully',
                    'timestamp': time.time()
                }
            else:
                return {
                    'type': 'error',
                    'message': 'Failed to stop training',
                    'timestamp': time.time()
                }
                
        except Exception as e:
            return {
                'type': 'error',
                'message': str(e),
                'timestamp': time.time()
            }
    
    async def _handle_update_reward_config(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle reward configuration update"""
        try:
            reward_config_data = data.get('reward_config', {})
            
            from .ml_interface import RewardConfig
            reward_config = RewardConfig.from_dict(reward_config_data)
            
            success = self.ml_interface.update_reward_config(reward_config)
            
            if success:
                # Update ROS bridge parameters
                self.ros_bridge.update_reward_parameters(reward_config.to_dict())
                
                # Broadcast update to all clients
                await self._broadcast_to_all({
                    'type': 'reward_config_updated',
                    'data': reward_config_data,
                    'timestamp': time.time()
                })
                
                return {
                    'type': 'reward_config_updated',
                    'message': 'Reward configuration updated successfully',
                    'timestamp': time.time()
                }
            else:
                return {
                    'type': 'error',
                    'message': 'Failed to update reward configuration',
                    'timestamp': time.time()
                }
                
        except Exception as e:
            return {
                'type': 'error',
                'message': str(e),
                'timestamp': time.time()
            }
    
    async def _handle_update_hyperparameters(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle hyperparameter update"""
        try:
            params = data.get('hyperparameters', {})
            
            success = self.ml_interface.update_hyperparameters(params)
            
            if success:
                # Broadcast update to all clients
                await self._broadcast_to_all({
                    'type': 'hyperparameters_updated',
                    'data': params,
                    'timestamp': time.time()
                })
                
                return {
                    'type': 'hyperparameters_updated',
                    'message': 'Hyperparameters updated successfully',
                    'timestamp': time.time()
                }
            else:
                return {
                    'type': 'error',
                    'message': 'Failed to update hyperparameters',
                    'timestamp': time.time()
                }
                
        except Exception as e:
            return {
                'type': 'error',
                'message': str(e),
                'timestamp': time.time()
            }
    
    async def _handle_get_status(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle status request"""
        try:
            status = self.ml_interface.get_system_status()
            metrics = self.ml_interface.get_live_training_metrics()
            progress = self.ml_interface.get_training_progress()
            
            return {
                'type': 'status_response',
                'data': {
                    'system_status': status,
                    'training_metrics': asdict(metrics),
                    'training_progress': progress,
                    'connected_clients': len(self.clients),
                    'ros_bridge_connected': self.ros_bridge.is_connected()
                },
                'timestamp': time.time()
            }
            
        except Exception as e:
            return {
                'type': 'error',
                'message': str(e),
                'timestamp': time.time()
            }
    
    async def _handle_ping(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle ping request"""
        return {
            'type': 'pong',
            'timestamp': time.time()
        }


# Global WebSocket manager instance
_websocket_manager: Optional[WebSocketManager] = None


def get_websocket_manager(host: str = "localhost", port: int = 8765) -> WebSocketManager:
    """Get global WebSocket manager instance"""
    global _websocket_manager
    if _websocket_manager is None:
        _websocket_manager = WebSocketManager(host, port)
    return _websocket_manager


async def start_websocket_server(host: str = "localhost", port: int = 8765) -> bool:
    """Start WebSocket server (call this when API server starts)"""
    manager = get_websocket_manager(host, port)
    return await manager.start_server()


async def stop_websocket_server():
    """Stop WebSocket server (call this when API server shuts down)"""
    global _websocket_manager
    if _websocket_manager:
        await _websocket_manager.stop_server()
        _websocket_manager = None


# Threading support for integration with existing Flask/FastAPI servers
class WebSocketThread:
    """Thread wrapper for running WebSocket server alongside existing API"""
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.loop = None
        self.thread = None
        self.manager = None
        
    def start(self) -> bool:
        """Start WebSocket server in background thread"""
        try:
            self.thread = threading.Thread(target=self._run_server, daemon=True)
            self.thread.start()
            
            # Wait a moment for server to start
            time.sleep(1.0)
            
            logger.info(f"WebSocket server thread started on ws://{self.host}:{self.port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start WebSocket server thread: {e}")
            return False
    
    def stop(self):
        """Stop WebSocket server"""
        if self.loop and self.manager:
            # Schedule stop in the event loop
            asyncio.run_coroutine_threadsafe(self.manager.stop_server(), self.loop)
            
        if self.thread:
            self.thread.join(timeout=5.0)
            
        logger.info("WebSocket server thread stopped")
    
    def _run_server(self):
        """Run server in thread"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        try:
            self.manager = WebSocketManager(self.host, self.port)
            self.loop.run_until_complete(self.manager.start_server())
            self.loop.run_forever()
        except Exception as e:
            logger.error(f"WebSocket server error: {e}")
        finally:
            self.loop.close()


# Convenience functions
def start_websocket_thread(host: str = "localhost", port: int = 8765) -> WebSocketThread:
    """Start WebSocket server in background thread"""
    ws_thread = WebSocketThread(host, port)
    ws_thread.start()
    return ws_thread
