#!/usr/bin/env python3
"""
ROS2 Bridge for Real-time ML Interface Communication
Enables Jay's backend to communicate with running ROS2 nodes
"""

import threading
import time
import queue
import logging
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, asdict

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from std_msgs.msg import String, Float32
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    Node = object

# Custom messages (when available)
try:
    from deepflyer_msgs.msg import RewardFeedback, CourseState, VisionFeatures
    CUSTOM_MSGS_AVAILABLE = True
except ImportError:
    CUSTOM_MSGS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class RealTimeData:
    """Real-time training data structure"""
    timestamp: float
    reward_total: float
    reward_breakdown: Dict[str, float]
    episode_progress: float
    hoops_completed: int
    current_hoop_id: int
    hoop_detected: bool
    hoop_distance: float
    training_loss: Optional[float] = None
    episode_number: Optional[int] = None


class ROSBridgeNode(Node):
    """ROS2 node that bridges ML interface with running components"""
    
    def __init__(self, data_callback: Callable[[RealTimeData], None]):
        super().__init__('ml_interface_bridge')
        
        self.data_callback = data_callback
        self.parameter_queue = queue.Queue()
        
        # QoS profiles
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Subscribers for real-time data
        if CUSTOM_MSGS_AVAILABLE:
            self.reward_sub = self.create_subscription(
                RewardFeedback, '/deepflyer/reward_feedback',
                self._reward_callback, reliable_qos
            )
            
            self.course_sub = self.create_subscription(
                CourseState, '/deepflyer/course_state',
                self._course_callback, reliable_qos
            )
            
            self.vision_sub = self.create_subscription(
                VisionFeatures, '/deepflyer/vision_features',
                self._vision_callback, reliable_qos
            )
        
        # Publishers for parameter updates
        self.param_update_pub = self.create_publisher(
            String, '/deepflyer/parameter_update', reliable_qos
        )
        
        # State tracking
        self.latest_reward: Optional[RewardFeedback] = None
        self.latest_course: Optional[CourseState] = None
        self.latest_vision: Optional[VisionFeatures] = None
        
        # Parameter update timer
        self.param_timer = self.create_timer(0.1, self._process_parameter_updates)
        
        # Data publishing timer
        self.data_timer = self.create_timer(0.2, self._publish_aggregated_data)
        
        logger.info("ROS Bridge node initialized")
    
    def _reward_callback(self, msg: RewardFeedback):
        """Handle reward feedback updates"""
        self.latest_reward = msg
    
    def _course_callback(self, msg: CourseState):
        """Handle course state updates"""
        self.latest_course = msg
    
    def _vision_callback(self, msg: VisionFeatures):
        """Handle vision feature updates"""
        self.latest_vision = msg
    
    def _publish_aggregated_data(self):
        """Aggregate and publish real-time data"""
        if not all([self.latest_reward, self.latest_course, self.latest_vision]):
            return
        
        data = RealTimeData(
            timestamp=time.time(),
            reward_total=self.latest_reward.total_reward,
            reward_breakdown={
                'hoop_progress': self.latest_reward.hoop_progress_reward,
                'alignment': self.latest_reward.alignment_reward,
                'collision_penalty': self.latest_reward.collision_penalty
            },
            episode_progress=self.latest_course.course_progress,
            hoops_completed=self.latest_course.hoops_completed,
            current_hoop_id=self.latest_course.current_target_hoop_id,
            hoop_detected=self.latest_vision.hoop_detected,
            hoop_distance=self.latest_vision.hoop_distance
        )
        
        # Send to callback
        if self.data_callback:
            self.data_callback(data)
    
    def _process_parameter_updates(self):
        """Process queued parameter updates"""
        try:
            while not self.parameter_queue.empty():
                update = self.parameter_queue.get_nowait()
                
                # Publish parameter update
                msg = String()
                msg.data = str(update)
                self.param_update_pub.publish(msg)
                
                logger.info(f"Published parameter update: {update}")
                
        except queue.Empty:
            pass
    
    def update_parameters(self, params: Dict[str, Any]):
        """Queue parameter update for ROS publication"""
        self.parameter_queue.put(params)
        logger.info(f"Queued parameter update: {params}")


class ROSMLBridge:
    """Main bridge class for ML interface integration"""
    
    def __init__(self):
        self.node: Optional[ROSBridgeNode] = None
        self.executor = None
        self.thread = None
        self.running = False
        
        self.data_callbacks = []
        self.latest_data: Optional[RealTimeData] = None
        
        if not ROS_AVAILABLE:
            logger.warning("ROS2 not available - running in mock mode")
    
    def start(self) -> bool:
        """Start the ROS bridge"""
        if not ROS_AVAILABLE:
            logger.warning("Cannot start ROS bridge - ROS2 not available")
            return False
        
        try:
            rclpy.init()
            self.node = ROSBridgeNode(self._handle_data_update)
            
            # Start ROS spinning in separate thread
            self.running = True
            self.thread = threading.Thread(target=self._spin_thread, daemon=True)
            self.thread.start()
            
            logger.info("ROS Bridge started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start ROS bridge: {e}")
            return False
    
    def stop(self):
        """Stop the ROS bridge"""
        self.running = False
        
        if self.thread:
            self.thread.join(timeout=5.0)
        
        if self.node:
            self.node.destroy_node()
        
        try:
            rclpy.shutdown()
        except:
            pass
        
        logger.info("ROS Bridge stopped")
    
    def _spin_thread(self):
        """ROS spinning thread"""
        try:
            while self.running and rclpy.ok():
                rclpy.spin_once(self.node, timeout_sec=0.1)
        except Exception as e:
            logger.error(f"ROS spin error: {e}")
    
    def _handle_data_update(self, data: RealTimeData):
        """Handle incoming real-time data"""
        self.latest_data = data
        
        # Notify all callbacks
        for callback in self.data_callbacks:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Data callback error: {e}")
    
    def add_data_callback(self, callback: Callable[[RealTimeData], None]):
        """Add callback for real-time data updates"""
        self.data_callbacks.append(callback)
    
    def update_reward_parameters(self, params: Dict[str, float]) -> bool:
        """Update reward parameters in real-time"""
        if not self.node:
            logger.warning("ROS bridge not running")
            return False
        
        try:
            self.node.update_parameters({
                'type': 'reward_params',
                'data': params
            })
            return True
        except Exception as e:
            logger.error(f"Failed to update parameters: {e}")
            return False
    
    def get_latest_data(self) -> Optional[Dict[str, Any]]:
        """Get latest training data"""
        if self.latest_data:
            return asdict(self.latest_data)
        return None
    
    def is_connected(self) -> bool:
        """Check if bridge is connected to ROS"""
        return self.running and self.node is not None


# Global bridge instance
_bridge_instance: Optional[ROSMLBridge] = None


def get_ros_bridge() -> ROSMLBridge:
    """Get global ROS bridge instance"""
    global _bridge_instance
    if _bridge_instance is None:
        _bridge_instance = ROSMLBridge()
    return _bridge_instance


# Convenience functions for ML interface
def start_ros_bridge() -> bool:
    """Start ROS bridge (call this when server starts)"""
    return get_ros_bridge().start()


def stop_ros_bridge():
    """Stop ROS bridge (call this when server shuts down)"""
    get_ros_bridge().stop()


def update_reward_parameters(params: Dict[str, float]) -> bool:
    """Update reward parameters in real-time"""
    return get_ros_bridge().update_reward_parameters(params)


def get_realtime_data() -> Optional[Dict[str, Any]]:
    """Get latest real-time training data"""
    return get_ros_bridge().get_latest_data()


def add_realtime_callback(callback: Callable[[RealTimeData], None]):
    """Add callback for real-time data updates"""
    get_ros_bridge().add_data_callback(callback) 