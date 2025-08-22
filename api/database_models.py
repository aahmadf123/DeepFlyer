#!/usr/bin/env python3
"""
SQLAlchemy Database Models for DeepFlyer ML Training
Enforces database schema validation and provides ORM interface
"""

import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
import json

try:
    from sqlalchemy import (
        Column, String, Integer, DateTime, Boolean, Text, 
        ForeignKey, CheckConstraint, UniqueConstraint, Index,
        Numeric, DECIMAL, JSON
    )
    from sqlalchemy.dialects.postgresql import UUID
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import relationship, validates
    from sqlalchemy.sql import func
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    # Create dummy base for when SQLAlchemy isn't available
    class declarative_base:
        def __init__(self):
            pass
    Column = String = Integer = DateTime = Boolean = Text = object
    ForeignKey = CheckConstraint = UniqueConstraint = Index = object
    Numeric = DECIMAL = JSON = UUID = relationship = validates = func = object

Base = declarative_base()


class TrainingSession(Base):
    """Training session model with validation constraints"""
    __tablename__ = 'training_sessions'
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False)  # References Jay's user table
    session_name = Column(String(100), nullable=False)
    
    # Training configuration
    training_minutes = Column(Integer, nullable=False)
    algorithm = Column(String(20), default='P3O')
    
    # Session status
    status = Column(String(20), default='created')
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    # Performance metrics
    total_episodes = Column(Integer, default=0)
    best_reward = Column(DECIMAL(10, 4), default=0.0)
    final_reward = Column(DECIMAL(10, 4), default=0.0)
    success_rate = Column(DECIMAL(5, 4), default=0.0)
    
    # Timestamps
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())
    
    # Relationships
    hyperparameter_configs = relationship("HyperparameterConfig", back_populates="session", cascade="all, delete-orphan")
    reward_configs = relationship("RewardConfig", back_populates="session", cascade="all, delete-orphan")
    training_metrics = relationship("TrainingMetric", back_populates="session", cascade="all, delete-orphan")
    hyperparameter_trials = relationship("HyperparameterTrial", back_populates="session", cascade="all, delete-orphan")
    model_checkpoints = relationship("ModelCheckpoint", back_populates="session", cascade="all, delete-orphan")
    optimization_suggestions = relationship("OptimizationSuggestion", back_populates="session", cascade="all, delete-orphan")
    
    # Constraints
    __table_args__ = (
        CheckConstraint('training_minutes >= 10 AND training_minutes <= 180', 
                       name='check_training_minutes_range'),
        CheckConstraint("status IN ('created', 'running', 'completed', 'failed', 'stopped')", 
                       name='check_status_values'),
        Index('idx_training_sessions_user_id', 'user_id'),
        Index('idx_training_sessions_status', 'status'),
        Index('idx_training_sessions_created_at', 'created_at'),
    )
    
    @validates('training_minutes')
    def validate_training_minutes(self, key, value):
        if not (10 <= value <= 180):
            raise ValueError(f"Training minutes must be between 10 and 180, got {value}")
        return value
    
    @validates('status')
    def validate_status(self, key, value):
        allowed_statuses = {'created', 'running', 'completed', 'failed', 'stopped'}
        if value not in allowed_statuses:
            raise ValueError(f"Status must be one of {allowed_statuses}, got {value}")
        return value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'id': str(self.id),
            'user_id': str(self.user_id),
            'session_name': self.session_name,
            'training_minutes': self.training_minutes,
            'algorithm': self.algorithm,
            'status': self.status,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'total_episodes': self.total_episodes,
            'best_reward': float(self.best_reward) if self.best_reward else 0.0,
            'final_reward': float(self.final_reward) if self.final_reward else 0.0,
            'success_rate': float(self.success_rate) if self.success_rate else 0.0,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class HyperparameterConfig(Base):
    """P3O hyperparameter configuration with student-tunable constraints"""
    __tablename__ = 'hyperparameter_configs'
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey('training_sessions.id', ondelete='CASCADE'), nullable=False)
    
    # P3O Hyperparameters (Student Configurable)
    learning_rate = Column(DECIMAL(10, 8), default=0.0003)
    clip_ratio = Column(DECIMAL(4, 3), default=0.2)
    entropy_coef = Column(DECIMAL(6, 5), default=0.01)
    batch_size = Column(Integer, default=64)
    rollout_steps = Column(Integer, default=512)
    num_epochs = Column(Integer, default=10)
    gamma = Column(DECIMAL(4, 3), default=0.99)
    gae_lambda = Column(DECIMAL(4, 3), default=0.95)
    
    # Configuration metadata
    is_default = Column(Boolean, default=False)
    created_at = Column(DateTime, default=func.current_timestamp())
    
    # Relationships
    session = relationship("TrainingSession", back_populates="hyperparameter_configs")
    
    # Constraints
    __table_args__ = (
        CheckConstraint('learning_rate >= 0.0001 AND learning_rate <= 0.003', 
                       name='check_learning_rate_range'),
        CheckConstraint('clip_ratio >= 0.1 AND clip_ratio <= 0.3', 
                       name='check_clip_ratio_range'),
        CheckConstraint('entropy_coef >= 0.001 AND entropy_coef <= 0.1', 
                       name='check_entropy_coef_range'),
        CheckConstraint('batch_size IN (64, 128, 256)', 
                       name='check_batch_size_values'),
        CheckConstraint('rollout_steps IN (512, 1024, 2048)', 
                       name='check_rollout_steps_values'),
        CheckConstraint('num_epochs >= 3 AND num_epochs <= 10', 
                       name='check_num_epochs_range'),
        CheckConstraint('gamma >= 0.9 AND gamma <= 0.99', 
                       name='check_gamma_range'),
        CheckConstraint('gae_lambda >= 0.9 AND gae_lambda <= 0.99', 
                       name='check_gae_lambda_range'),
    )
    
    @validates('learning_rate')
    def validate_learning_rate(self, key, value):
        if not (0.0001 <= float(value) <= 0.003):
            raise ValueError(f"Learning rate must be between 0.0001 and 0.003, got {value}")
        return value
    
    @validates('clip_ratio')
    def validate_clip_ratio(self, key, value):
        if not (0.1 <= float(value) <= 0.3):
            raise ValueError(f"Clip ratio must be between 0.1 and 0.3, got {value}")
        return value
    
    @validates('batch_size')
    def validate_batch_size(self, key, value):
        if value not in [64, 128, 256]:
            raise ValueError(f"Batch size must be one of [64, 128, 256], got {value}")
        return value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'id': str(self.id),
            'session_id': str(self.session_id),
            'learning_rate': float(self.learning_rate),
            'clip_ratio': float(self.clip_ratio),
            'entropy_coef': float(self.entropy_coef),
            'batch_size': self.batch_size,
            'rollout_steps': self.rollout_steps,
            'num_epochs': self.num_epochs,
            'gamma': float(self.gamma),
            'gae_lambda': float(self.gae_lambda),
            'is_default': self.is_default,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class RewardConfig(Base):
    """Reward function configuration with student-tunable parameters"""
    __tablename__ = 'reward_configs'
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey('training_sessions.id', ondelete='CASCADE'), nullable=False)
    
    # Positive rewards (student tunable)
    hoop_approach_reward = Column(DECIMAL(8, 2), default=10.0)
    hoop_passage_reward = Column(DECIMAL(8, 2), default=50.0)
    visual_alignment_reward = Column(DECIMAL(8, 2), default=5.0)
    forward_progress_reward = Column(DECIMAL(8, 2), default=3.0)
    
    # Penalties (student tunable)
    wrong_direction_penalty = Column(DECIMAL(8, 2), default=-2.0)
    hoop_miss_penalty = Column(DECIMAL(8, 2), default=-25.0)
    collision_penalty = Column(DECIMAL(8, 2), default=-100.0)
    
    # Reward function code
    reward_function_code = Column(Text)
    
    # Metadata
    created_at = Column(DateTime, default=func.current_timestamp())
    
    # Relationships
    session = relationship("TrainingSession", back_populates="reward_configs")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'id': str(self.id),
            'session_id': str(self.session_id),
            'hoop_approach_reward': float(self.hoop_approach_reward),
            'hoop_passage_reward': float(self.hoop_passage_reward),
            'visual_alignment_reward': float(self.visual_alignment_reward),
            'forward_progress_reward': float(self.forward_progress_reward),
            'wrong_direction_penalty': float(self.wrong_direction_penalty),
            'hoop_miss_penalty': float(self.hoop_miss_penalty),
            'collision_penalty': float(self.collision_penalty),
            'reward_function_code': self.reward_function_code,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class TrainingMetric(Base):
    """Real-time training metrics from ClearML"""
    __tablename__ = 'training_metrics'
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey('training_sessions.id', ondelete='CASCADE'), nullable=False)
    
    # Episode information
    episode_number = Column(Integer, nullable=False)
    episode_reward = Column(DECIMAL(10, 4), nullable=False)
    episode_length = Column(Integer, nullable=False)
    
    # Learning metrics
    policy_loss = Column(DECIMAL(10, 6))
    value_loss = Column(DECIMAL(10, 6))
    entropy = Column(DECIMAL(10, 6))
    
    # Task-specific metrics
    hoop_completion_rate = Column(DECIMAL(5, 4))
    collision_rate = Column(DECIMAL(5, 4))
    average_lap_time = Column(DECIMAL(8, 2))
    
    # Timestamp
    recorded_at = Column(DateTime, default=func.current_timestamp())
    
    # Relationships
    session = relationship("TrainingSession", back_populates="training_metrics")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('session_id', 'episode_number', name='uq_session_episode'),
        Index('idx_training_metrics_session_id', 'session_id'),
        Index('idx_training_metrics_episode', 'session_id', 'episode_number'),
        Index('idx_training_metrics_recorded_at', 'recorded_at'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'id': str(self.id),
            'session_id': str(self.session_id),
            'episode_number': self.episode_number,
            'episode_reward': float(self.episode_reward),
            'episode_length': self.episode_length,
            'policy_loss': float(self.policy_loss) if self.policy_loss else None,
            'value_loss': float(self.value_loss) if self.value_loss else None,
            'entropy': float(self.entropy) if self.entropy else None,
            'hoop_completion_rate': float(self.hoop_completion_rate) if self.hoop_completion_rate else None,
            'collision_rate': float(self.collision_rate) if self.collision_rate else None,
            'average_lap_time': float(self.average_lap_time) if self.average_lap_time else None,
            'recorded_at': self.recorded_at.isoformat() if self.recorded_at else None
        }


class HyperparameterTrial(Base):
    """Hyperparameter optimization trial results"""
    __tablename__ = 'hyperparameter_trials'
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey('training_sessions.id', ondelete='CASCADE'), nullable=False)
    
    # Trial information
    trial_number = Column(Integer, nullable=False)
    trial_config = Column(JSON, nullable=False)  # Store hyperparameter configuration as JSON
    
    # Performance results
    performance_score = Column(DECIMAL(10, 4))
    final_reward = Column(DECIMAL(10, 4))
    success_rate = Column(DECIMAL(5, 4))
    
    # Trial status
    status = Column(String(20), default='running')
    started_at = Column(DateTime, default=func.current_timestamp())
    completed_at = Column(DateTime)
    duration_seconds = Column(Integer)
    
    # Relationships
    session = relationship("TrainingSession", back_populates="hyperparameter_trials")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('session_id', 'trial_number', name='uq_session_trial'),
        CheckConstraint("status IN ('running', 'completed', 'failed')", 
                       name='check_trial_status_values'),
        Index('idx_hyperparameter_trials_session_id', 'session_id'),
        Index('idx_hyperparameter_trials_performance', 'performance_score'),
    )
    
    @validates('status')
    def validate_status(self, key, value):
        allowed_statuses = {'running', 'completed', 'failed'}
        if value not in allowed_statuses:
            raise ValueError(f"Trial status must be one of {allowed_statuses}, got {value}")
        return value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'id': str(self.id),
            'session_id': str(self.session_id),
            'trial_number': self.trial_number,
            'trial_config': self.trial_config,
            'performance_score': float(self.performance_score) if self.performance_score else None,
            'final_reward': float(self.final_reward) if self.final_reward else None,
            'success_rate': float(self.success_rate) if self.success_rate else None,
            'status': self.status,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'duration_seconds': self.duration_seconds
        }


class ModelCheckpoint(Base):
    """Model checkpoint storage metadata"""
    __tablename__ = 'model_checkpoints'
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey('training_sessions.id', ondelete='CASCADE'), nullable=False)
    
    # Checkpoint information
    checkpoint_type = Column(String(20), nullable=False)
    episode_number = Column(Integer, nullable=False)
    reward_score = Column(DECIMAL(10, 4), nullable=False)
    
    # Storage information
    file_path = Column(String(500), nullable=False)
    file_size_bytes = Column(Integer)
    clearml_artifact_id = Column(String(200))
    
    # Checkpoint metadata
    training_time_elapsed = Column(Integer)  # seconds
    created_at = Column(DateTime, default=func.current_timestamp())
    
    # Relationships
    session = relationship("TrainingSession", back_populates="model_checkpoints")
    
    # Constraints
    __table_args__ = (
        CheckConstraint("checkpoint_type IN ('periodic', 'best', 'final')", 
                       name='check_checkpoint_type_values'),
        Index('idx_model_checkpoints_session_id', 'session_id'),
        Index('idx_model_checkpoints_type', 'checkpoint_type'),
        Index('idx_model_checkpoints_reward', 'reward_score'),
    )
    
    @validates('checkpoint_type')
    def validate_checkpoint_type(self, key, value):
        allowed_types = {'periodic', 'best', 'final'}
        if value not in allowed_types:
            raise ValueError(f"Checkpoint type must be one of {allowed_types}, got {value}")
        return value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'id': str(self.id),
            'session_id': str(self.session_id),
            'checkpoint_type': self.checkpoint_type,
            'episode_number': self.episode_number,
            'reward_score': float(self.reward_score),
            'file_path': self.file_path,
            'file_size_bytes': self.file_size_bytes,
            'clearml_artifact_id': self.clearml_artifact_id,
            'training_time_elapsed': self.training_time_elapsed,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class OptimizationSuggestion(Base):
    """AI-generated optimization suggestions"""
    __tablename__ = 'optimization_suggestions'
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey('training_sessions.id', ondelete='CASCADE'), nullable=False)
    
    # Suggestion content
    suggestion_text = Column(Text, nullable=False)
    suggestion_type = Column(String(30), nullable=False)
    confidence_score = Column(DECIMAL(3, 2))
    
    # Context
    based_on_trials = Column(Integer)  # Number of trials this suggestion is based on
    generated_at = Column(DateTime, default=func.current_timestamp())
    
    # Student interaction
    accepted = Column(Boolean)
    applied_at = Column(DateTime)
    
    # Relationships
    session = relationship("TrainingSession", back_populates="optimization_suggestions")
    
    # Constraints
    __table_args__ = (
        CheckConstraint("suggestion_type IN ('hyperparameter', 'reward', 'training_time', 'general')", 
                       name='check_suggestion_type_values'),
        CheckConstraint('confidence_score >= 0.0 AND confidence_score <= 1.0', 
                       name='check_confidence_score_range'),
    )
    
    @validates('suggestion_type')
    def validate_suggestion_type(self, key, value):
        allowed_types = {'hyperparameter', 'reward', 'training_time', 'general'}
        if value not in allowed_types:
            raise ValueError(f"Suggestion type must be one of {allowed_types}, got {value}")
        return value
    
    @validates('confidence_score')
    def validate_confidence_score(self, key, value):
        if value is not None and not (0.0 <= float(value) <= 1.0):
            raise ValueError(f"Confidence score must be between 0.0 and 1.0, got {value}")
        return value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'id': str(self.id),
            'session_id': str(self.session_id),
            'suggestion_text': self.suggestion_text,
            'suggestion_type': self.suggestion_type,
            'confidence_score': float(self.confidence_score) if self.confidence_score else None,
            'based_on_trials': self.based_on_trials,
            'generated_at': self.generated_at.isoformat() if self.generated_at else None,
            'accepted': self.accepted,
            'applied_at': self.applied_at.isoformat() if self.applied_at else None
        }


# Database utilities and helper functions

class DatabaseManager:
    """Database connection and validation manager"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = None
        self.session_factory = None
        
    def initialize(self):
        """Initialize database connection and create tables"""
        if not SQLALCHEMY_AVAILABLE:
            raise ImportError("SQLAlchemy not available. Install with: pip install SQLAlchemy psycopg2-binary")
        
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        
        # Create engine
        self.engine = create_engine(
            self.database_url,
            pool_pre_ping=True,
            pool_recycle=300,
            echo=False  # Set to True for SQL debugging
        )
        
        # Create session factory
        self.session_factory = sessionmaker(bind=self.engine)
        
        # Create all tables
        Base.metadata.create_all(self.engine)
        
    def get_session(self):
        """Get database session"""
        if not self.session_factory:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self.session_factory()
    
    def validate_training_session(self, session_data: Dict[str, Any]) -> bool:
        """Validate training session data against constraints"""
        try:
            # Create temporary session object for validation
            temp_session = TrainingSession(**session_data)
            return True
        except (ValueError, TypeError) as e:
            print(f"Validation error: {e}")
            return False
    
    def validate_hyperparameters(self, hyperparams: Dict[str, Any]) -> bool:
        """Validate hyperparameter configuration"""
        try:
            # Create temporary config object for validation
            temp_config = HyperparameterConfig(**hyperparams)
            return True
        except (ValueError, TypeError) as e:
            print(f"Hyperparameter validation error: {e}")
            return False


# Global database manager
_db_manager: Optional[DatabaseManager] = None


def get_database_manager(database_url: str = None) -> DatabaseManager:
    """Get global database manager instance"""
    global _db_manager
    if _db_manager is None and database_url:
        _db_manager = DatabaseManager(database_url)
    return _db_manager


def initialize_database(database_url: str) -> DatabaseManager:
    """Initialize database with the given URL"""
    manager = DatabaseManager(database_url)
    manager.initialize()
    global _db_manager
    _db_manager = manager
    return manager


# Validation functions for API integration

def validate_training_request(data: Dict[str, Any]) -> List[str]:
    """
    Validate training request data
    
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Check required fields
    required_fields = ['training_minutes', 'session_name']
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")
    
    # Validate training minutes
    training_minutes = data.get('training_minutes')
    if training_minutes is not None:
        if not isinstance(training_minutes, int) or not (10 <= training_minutes <= 180):
            errors.append("training_minutes must be an integer between 10 and 180")
    
    # Validate session name
    session_name = data.get('session_name')
    if session_name is not None:
        if not isinstance(session_name, str) or len(session_name) > 100:
            errors.append("session_name must be a string with max 100 characters")
    
    # Validate hyperparameters if provided
    hyperparams = data.get('hyperparameters', {})
    if hyperparams:
        hp_errors = validate_hyperparameters_dict(hyperparams)
        errors.extend(hp_errors)
    
    return errors


def validate_hyperparameters_dict(hyperparams: Dict[str, Any]) -> List[str]:
    """Validate hyperparameter dictionary"""
    errors = []
    
    # Learning rate validation
    lr = hyperparams.get('learning_rate')
    if lr is not None:
        try:
            lr_float = float(lr)
            if not (0.0001 <= lr_float <= 0.003):
                errors.append("learning_rate must be between 0.0001 and 0.003")
        except (ValueError, TypeError):
            errors.append("learning_rate must be a number")
    
    # Batch size validation
    batch_size = hyperparams.get('batch_size')
    if batch_size is not None:
        if batch_size not in [64, 128, 256]:
            errors.append("batch_size must be one of [64, 128, 256]")
    
    # Clip ratio validation
    clip_ratio = hyperparams.get('clip_ratio')
    if clip_ratio is not None:
        try:
            clip_float = float(clip_ratio)
            if not (0.1 <= clip_float <= 0.3):
                errors.append("clip_ratio must be between 0.1 and 0.3")
        except (ValueError, TypeError):
            errors.append("clip_ratio must be a number")
    
    return errors


def validate_reward_config_dict(reward_config: Dict[str, Any]) -> List[str]:
    """Validate reward configuration dictionary"""
    errors = []
    
    # All reward values should be numeric
    numeric_fields = [
        'hoop_approach_reward', 'hoop_passage_reward', 'visual_alignment_reward',
        'forward_progress_reward', 'wrong_direction_penalty', 'hoop_miss_penalty',
        'collision_penalty'
    ]
    
    for field in numeric_fields:
        value = reward_config.get(field)
        if value is not None:
            try:
                float(value)
            except (ValueError, TypeError):
                errors.append(f"{field} must be a number")
    
    return errors
