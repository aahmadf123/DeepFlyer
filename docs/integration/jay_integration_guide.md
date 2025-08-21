# DeepFlyer Backend Integration Documentation
**For Backend, UI, Database, and Frontend Development**

## Overview
This document describes the available ML interfaces, data structures, and integration points for building a backend system that connects to the DeepFlyer RL training platform. The UI follows AWS DeepRacer's interface pattern.

### Related Documentation
- **📚 [Technical Overview](../TEAM_OVERVIEW.md)** - Complete ML/RL system implementation details
- **🚁 [Simulation Integration](../UMA_INTEGRATION_GUIDE.md)** - Uma's simulation interface specifications
- **📖 [System Architecture](../INTEGRATION_GUIDE.md)** - High-level integration overview

## Key Files You Need to Know About

### My ML Interface (Your Main Integration Point)
```
api/ml_interface.py         # MAIN: DeepFlyerMLInterface class
api/neon_database_schema.sql # Database schema suggestion
rl_agent/config.py          # All hyperparameters and settings
rl_agent/algorithms/p3o.py  # P3O algorithm and hyperparameter optimization
rl_agent/rewards/rewards.py # DeepRacer-style reward function
```

### Configuration and Data Structures
```
rl_agent/config.py          # All default values for UI
rl_agent/algorithms/p3o.py (lines 37-141) # Student-tunable hyperparameters
rl_agent/rewards/rewards.py # Student reward function template
```

## Main Integration Interface

### DeepFlyerMLInterface Class (api/ml_interface.py)
This is your main integration point. Use it like this:

```python
from api.ml_interface import DeepFlyerMLInterface, RewardConfig

# Initialize ML interface
ml = DeepFlyerMLInterface()

# Start training (REQUIRED: students must specify training time)
success = ml.start_training(
    training_minutes=60,  # REQUIRED - no default
    reward_config=RewardConfig(...),
    hyperparameters={'learning_rate': 1e-3}
)

# Get live metrics for dashboard (call every 2-3 seconds)
metrics = ml.get_live_training_metrics()

# Get progress for progress bars
progress = ml.get_training_progress()

# Get reward breakdown
rewards = ml.get_reward_breakdown()
```

## Student-Configurable Parameters

### 1. Training Time (CRITICAL REQUIREMENT)
**From:** `rl_agent/algorithms/p3o.py` (lines 160-200)

```python
# Students MUST specify training time - NO DEFAULT VALUE
train_time_minutes = None  # REQUIRED: 10-180 minutes
max_steps = train_time_minutes * 60 * 20  # 20Hz control rate
```

**UI Requirements:**
- Required field, no default
- Range: 10-180 minutes 
- Show estimated compute cost
- Form validation prevents training without this

### 2. P3O Hyperparameters (Student Tunable)
**From:** `rl_agent/algorithms/p3o.py` (lines 37-141)

```python
# Available in p3o_config.get_student_config():
{
    'learning_rate': {
        'value': 3e-4,
        'range': [1e-4, 3e-3],
        'type': 'float',
        'scale': 'log'
    },
    'clip_ratio': {
        'value': 0.2,
        'range': [0.1, 0.3],
        'type': 'float'
    },
    'batch_size': {
        'value': 64,
        'options': [64, 128, 256],
        'type': 'select'
    }
    # ... more parameters
}
```

### 3. Reward Function (DeepRacer Style)
**From:** `rl_agent/rewards/rewards.py`

Students edit Python code directly:

```python
def reward_function(params):
    # Student-modifiable parameters
    HOOP_APPROACH_REWARD = 10.0      # Can modify
    HOOP_PASSAGE_REWARD = 50.0       # Can modify
    COLLISION_PENALTY = -100.0       # Can modify
    
    # Students can completely rewrite this function
    total_reward = 0.0
    if params.get('approaching_hoop', False):
        total_reward += HOOP_APPROACH_REWARD
    # ... rest of function
    return total_reward
```

## Real-Time Data Interface

### 1. Live Training Metrics
**Call:** `ml.get_live_training_metrics()` every 2-3 seconds

```python
# Returns TrainingMetrics object with:
{
    'is_training': True,
    'current_episode': 45,
    'current_reward': 33.2,
    'average_reward': 28.7,
    'best_reward': 89.1,
    'policy_loss': 0.023,
    'value_loss': 0.156,
    'hoop_completion_rate': 0.67,
    'collision_rate': 0.12,
    'training_time_elapsed': 180.5
}
```

### 2. Progress Information  
**Call:** `ml.get_training_progress()`

```python
# Returns progress info for progress bars:
{
    'episode_progress': 45.0,        # Percentage by episodes
    'time_progress': 30.1,           # Percentage by time
    'status': 'training',            # 'training' or 'stopped'
    'current_episode': 45,
    'estimated_remaining': 420.3     # Seconds remaining
}
```

### 3. Reward Breakdown
**Call:** `ml.get_reward_breakdown()`

```python
# Returns reward components:
{
    'hoop_approach': 5.2,
    'hoop_passage': 50.0,
    'visual_alignment': 3.1,
    'collision_penalty': -25.0,
    'total_reward': 33.3
}
```

## Hyperparameter Optimization (Like AWS DeepRacer)

### Starting Optimization
```python
# Start random search optimization
ml.start_hyperparameter_optimization(num_trials=20)

# Get all trial results
trials = ml.get_optimization_trials()
# Returns list of HyperparameterTrial objects

# Get best configuration
best_config = ml.get_best_hyperparameters()

# Get AI suggestions
suggestions = ml.get_optimization_suggestions()
# Returns: ["Try higher learning rates", "Increase batch size", ...]
```

### Trial Results Structure
```python
# Each trial has:
{
    'trial_number': 1,
    'hyperparameters': {
        'learning_rate': 0.001,
        'clip_ratio': 0.2,
        'batch_size': 128
    },
    'performance': 45.6,
    'status': 'completed',
    'duration': 300
}
```

## Database Schema (Your Implementation)

### Core Tables (from `api/neon_database_schema.sql`)
```sql
-- Training sessions
CREATE TABLE training_sessions (
    session_id UUID PRIMARY KEY,
    user_id UUID NOT NULL,
    session_name VARCHAR(100) NOT NULL,
    training_minutes INTEGER NOT NULL,  -- REQUIRED field
    algorithm VARCHAR(50) DEFAULT 'P3O',
    created_at TIMESTAMP DEFAULT NOW()
);

-- Real-time metrics
CREATE TABLE training_metrics (
    session_id UUID REFERENCES training_sessions(session_id),
    episode_number INTEGER,
    reward DECIMAL(10,4),
    policy_loss DECIMAL(12,8),
    -- ... more metrics
);

-- Hyperparameter trials
CREATE TABLE hyperparameter_trials (
    trial_id UUID PRIMARY KEY,
    session_id UUID REFERENCES training_sessions(session_id),
    trial_number INTEGER,
    hyperparameters JSONB,
    performance DECIMAL(10,4)
);
```

## UI Requirements (AWS DeepRacer Style)

### 1. Session Creation Screen
```
Required elements:
- Session name input
- Training time selector (10-180 minutes, REQUIRED)
- Quick start options (30min/60min/120min)
- Estimated cost display
- Cannot start without training time
```

### 2. Code Editor (Python)
```
- Monaco Editor or CodeMirror
- Python syntax highlighting  
- Real-time syntax validation
- Auto-completion for reward parameters
- Template reward functions
- Save/load functionality
```

### 3. Hyperparameter Panel
```
- Sliders for continuous parameters
- Dropdowns for discrete choices
- Real-time validation
- Reset to defaults button
- Parameter descriptions
```

### 4. Live Dashboard
```
- Real-time charts (Chart.js/D3.js)
- Progress bars (episode + time)
- Performance metrics cards
- WebSocket or polling updates (2-3 seconds)
- Training control buttons
```

### 5. Optimization Dashboard
```
- Trial results table
- Performance comparison charts
- Best configuration display
- AI suggestions panel
- Apply configuration button
```

## API Endpoints You Need to Build

### Training Control
```python
POST /api/sessions/create
{
    "session_name": "My Training",
    "training_minutes": 60  # REQUIRED
}

POST /api/training/start
{
    "session_id": "uuid",
    "reward_config": {...},
    "hyperparameters": {...}
}

POST /api/training/stop
GET /api/training/status
```

### Live Data
```python
GET /api/sessions/{id}/metrics      # Live training metrics
GET /api/training/progress          # Progress percentages  
GET /api/training/rewards           # Reward breakdown
```

### Hyperparameter Optimization
```python
POST /api/hyperopt/start           # Start optimization
GET /api/hyperopt/trials           # Get trial results
GET /api/hyperopt/best             # Get best config
GET /api/hyperopt/suggestions      # Get AI suggestions
```

## Configuration Access

### Default Values for UI
```python
from rl_agent.config import DeepFlyerConfig
from rl_agent.algorithms.p3o import P3OConfig

# Get default hyperparameters
p3o_config = P3OConfig()
defaults = p3o_config.get_student_config()

# Get course configuration
course_config = DeepFlyerConfig.COURSE_DIMENSIONS
reward_config = DeepFlyerConfig.REWARD_CONFIG
```

### Validation Functions
```python
from rl_agent.algorithms.p3o import validate_student_config

# Validate student input
clean_config = validate_student_config({
    'learning_rate': '0.001',  # String from UI
    'batch_size': '128'
})
# Returns validated dict with proper types
```

## ClearML Integration (Automatic)

My ML interface automatically handles:
- Live metric collection from ClearML
- Hyperparameter logging
- Trial tracking
- Performance monitoring

You just call the interface methods - ClearML integration is handled internally.

## System Status Monitoring

```python
# Check system health
status = ml.get_system_status()
# Returns:
{
    'clearml_connected': True,
    'training_active': True, 
    'hyperopt_active': False,
    'last_update': '2024-01-15T10:30:00',
    'warnings': []
}
```

## Critical Requirements

### 1. Training Time
- NO DEFAULT VALUE - students must specify
- Range: 10-180 minutes
- Form validation required
- Show in session creation UI

### 2. Real-Time Updates
- Dashboard updates every 2-3 seconds
- Use WebSocket or polling
- Handle connection failures gracefully
- Show loading states

### 3. AWS DeepRacer UI Style
- Direct Python code editing
- Live training feedback
- Hyperparameter sliders
- Real-time charts
- Episode progress tracking

## System Components

### Provided by RL System
- RL training algorithms and P3O implementation
- ClearML integration and metrics collection
- Reward function execution and processing
- Real-time training data and optimization results

### Integration Requirements
1. **Backend API:** REST endpoints implementing the ML interface
2. **Database:** Session storage, metrics tracking, and trial history
3. **UI Components:** AWS DeepRacer-style interface elements
4. **Real-time updates:** Live dashboard with streaming metrics
5. **Code editor:** Python syntax highlighting and validation
6. **Optimization interface:** Hyperparameter trial visualization and management

## Implementation Checklist

- [ ] Database implementation using provided schema
- [ ] API endpoint development using ML interface
- [ ] Session creation UI with required training time field
- [ ] Python code editor implementation for reward function editing
- [ ] Hyperparameter control interface using configuration specifications
- [ ] Real-time dashboard with live metrics integration
- [ ] Optimization trial visualization and management interface
- [ ] Complete system integration testing with ML interface

## Reference Documentation

For detailed implementation specifications, reference:
- `api/ml_interface.py` - Primary integration interface
- `rl_agent/config.py` - Default values and system settings
- `rl_agent/algorithms/p3o.py` - Hyperparameter specifications
- `rl_agent/rewards/rewards.py` - Student reward function template

The user interface provides the educational platform for student interaction with the autonomous flight training system. 