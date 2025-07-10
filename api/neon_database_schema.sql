-- DeepFlyer ML Training Database Schema
-- Schema suggestion for integration with Jay's existing backend

-- ============================================================================
-- TRAINING SESSIONS
-- ============================================================================

CREATE TABLE training_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,  -- References Jay's existing user table
    session_name VARCHAR(100) NOT NULL,
    
    -- Training configuration
    training_minutes INTEGER NOT NULL CHECK (training_minutes >= 10 AND training_minutes <= 180),
    algorithm VARCHAR(20) DEFAULT 'P3O',
    
    -- Session status
    status VARCHAR(20) DEFAULT 'created' CHECK (status IN ('created', 'running', 'completed', 'failed', 'stopped')),
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    
    -- Performance metrics
    total_episodes INTEGER DEFAULT 0,
    best_reward DECIMAL(10,4) DEFAULT 0.0,
    final_reward DECIMAL(10,4) DEFAULT 0.0,
    success_rate DECIMAL(5,4) DEFAULT 0.0,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- HYPERPARAMETER CONFIGURATIONS
-- ============================================================================

CREATE TABLE hyperparameter_configs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES training_sessions(id) ON DELETE CASCADE,
    
    -- P3O Hyperparameters (Student Configurable)
    learning_rate DECIMAL(10,8) DEFAULT 0.0003 CHECK (learning_rate >= 0.0001 AND learning_rate <= 0.003),
    clip_ratio DECIMAL(4,3) DEFAULT 0.2 CHECK (clip_ratio >= 0.1 AND clip_ratio <= 0.3),
    entropy_coef DECIMAL(6,5) DEFAULT 0.01 CHECK (entropy_coef >= 0.001 AND entropy_coef <= 0.1),
    batch_size INTEGER DEFAULT 64 CHECK (batch_size IN (64, 128, 256)),
    rollout_steps INTEGER DEFAULT 512 CHECK (rollout_steps IN (512, 1024, 2048)),
    num_epochs INTEGER DEFAULT 10 CHECK (num_epochs >= 3 AND num_epochs <= 10),
    gamma DECIMAL(4,3) DEFAULT 0.99 CHECK (gamma >= 0.9 AND gamma <= 0.99),
    gae_lambda DECIMAL(4,3) DEFAULT 0.95 CHECK (gae_lambda >= 0.9 AND gae_lambda <= 0.99),
    
    -- Configuration metadata
    is_default BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- REWARD FUNCTION CONFIGURATIONS
-- ============================================================================

CREATE TABLE reward_configs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES training_sessions(id) ON DELETE CASCADE,
    
    -- Positive rewards (student tunable)
    hoop_approach_reward DECIMAL(8,2) DEFAULT 10.0,
    hoop_passage_reward DECIMAL(8,2) DEFAULT 50.0,
    visual_alignment_reward DECIMAL(8,2) DEFAULT 5.0,
    forward_progress_reward DECIMAL(8,2) DEFAULT 3.0,
    
    -- Penalties (student tunable)
    wrong_direction_penalty DECIMAL(8,2) DEFAULT -2.0,
    hoop_miss_penalty DECIMAL(8,2) DEFAULT -25.0,
    collision_penalty DECIMAL(8,2) DEFAULT -100.0,
    
    -- Reward function code
    reward_function_code TEXT,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- TRAINING METRICS (Real-time data from ClearML)
-- ============================================================================

CREATE TABLE training_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES training_sessions(id) ON DELETE CASCADE,
    
    -- Episode information
    episode_number INTEGER NOT NULL,
    episode_reward DECIMAL(10,4) NOT NULL,
    episode_length INTEGER NOT NULL,
    
    -- Learning metrics
    policy_loss DECIMAL(10,6),
    value_loss DECIMAL(10,6),
    entropy DECIMAL(10,6),
    
    -- Task-specific metrics
    hoop_completion_rate DECIMAL(5,4),
    collision_rate DECIMAL(5,4),
    average_lap_time DECIMAL(8,2),
    
    -- Timestamp
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Unique constraint to prevent duplicate episodes
    UNIQUE(session_id, episode_number)
);

-- ============================================================================
-- HYPERPARAMETER OPTIMIZATION TRIALS
-- ============================================================================

CREATE TABLE hyperparameter_trials (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES training_sessions(id) ON DELETE CASCADE,
    
    -- Trial information
    trial_number INTEGER NOT NULL,
    trial_config JSONB NOT NULL, -- Store hyperparameter configuration as JSON
    
    -- Performance results
    performance_score DECIMAL(10,4),
    final_reward DECIMAL(10,4),
    success_rate DECIMAL(5,4),
    
    -- Trial status
    status VARCHAR(20) DEFAULT 'running' CHECK (status IN ('running', 'completed', 'failed')),
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    duration_seconds INTEGER,
    
    -- Unique constraint
    UNIQUE(session_id, trial_number)
);

-- ============================================================================
-- MODEL CHECKPOINTS
-- ============================================================================

CREATE TABLE model_checkpoints (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES training_sessions(id) ON DELETE CASCADE,
    
    -- Checkpoint information
    checkpoint_type VARCHAR(20) NOT NULL CHECK (checkpoint_type IN ('periodic', 'best', 'final')),
    episode_number INTEGER NOT NULL,
    reward_score DECIMAL(10,4) NOT NULL,
    
    -- Storage information
    file_path VARCHAR(500) NOT NULL,
    file_size_bytes BIGINT,
    clearml_artifact_id VARCHAR(200),
    
    -- Checkpoint metadata
    training_time_elapsed INTEGER, -- seconds
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- OPTIMIZATION SUGGESTIONS (AI-generated)
-- ============================================================================

CREATE TABLE optimization_suggestions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES training_sessions(id) ON DELETE CASCADE,
    
    -- Suggestion content
    suggestion_text TEXT NOT NULL,
    suggestion_type VARCHAR(30) NOT NULL CHECK (suggestion_type IN ('hyperparameter', 'reward', 'training_time', 'general')),
    confidence_score DECIMAL(3,2) CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
    
    -- Context
    based_on_trials INTEGER, -- Number of trials this suggestion is based on
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Student interaction
    accepted BOOLEAN DEFAULT NULL,
    applied_at TIMESTAMP
);

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================

CREATE INDEX idx_training_sessions_user_id ON training_sessions(user_id);
CREATE INDEX idx_training_sessions_status ON training_sessions(status);
CREATE INDEX idx_training_sessions_created_at ON training_sessions(created_at);

CREATE INDEX idx_training_metrics_session_id ON training_metrics(session_id);
CREATE INDEX idx_training_metrics_episode ON training_metrics(session_id, episode_number);
CREATE INDEX idx_training_metrics_recorded_at ON training_metrics(recorded_at);

CREATE INDEX idx_hyperparameter_trials_session_id ON hyperparameter_trials(session_id);
CREATE INDEX idx_hyperparameter_trials_performance ON hyperparameter_trials(performance_score DESC);

CREATE INDEX idx_model_checkpoints_session_id ON model_checkpoints(session_id);
CREATE INDEX idx_model_checkpoints_type ON model_checkpoints(checkpoint_type);
CREATE INDEX idx_model_checkpoints_reward ON model_checkpoints(reward_score DESC);

-- ============================================================================
-- VIEWS FOR COMMON QUERIES
-- ============================================================================

-- View for current training session status
CREATE VIEW current_training_status AS
SELECT 
    ts.id,
    ts.user_id,
    ts.session_name,
    ts.status,
    ts.training_minutes,
    ts.started_at,
    ts.total_episodes,
    ts.best_reward,
    ts.success_rate,
    EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - ts.started_at))/60 AS elapsed_minutes
FROM training_sessions ts
WHERE ts.status IN ('running', 'created');

-- View for session performance summary
CREATE VIEW session_performance_summary AS
SELECT 
    ts.id,
    ts.session_name,
    ts.status,
    ts.training_minutes,
    ts.total_episodes,
    ts.best_reward,
    ts.success_rate,
    COUNT(tm.id) as recorded_episodes,
    AVG(tm.episode_reward) as avg_reward,
    MAX(tm.episode_reward) as max_reward,
    COUNT(mc.id) as checkpoint_count
FROM training_sessions ts
LEFT JOIN training_metrics tm ON ts.id = tm.session_id
LEFT JOIN model_checkpoints mc ON ts.id = mc.session_id
GROUP BY ts.id, ts.session_name, ts.status, ts.training_minutes, ts.total_episodes, ts.best_reward, ts.success_rate;

-- View for hyperparameter optimization results
CREATE VIEW hyperopt_results AS
SELECT 
    ht.session_id,
    ht.trial_number,
    ht.trial_config,
    ht.performance_score,
    ht.success_rate,
    ht.status,
    ht.duration_seconds,
    ROW_NUMBER() OVER (PARTITION BY ht.session_id ORDER BY ht.performance_score DESC) as performance_rank
FROM hyperparameter_trials ht
WHERE ht.status = 'completed';

-- ============================================================================
-- FUNCTIONS FOR ML INTEGRATION
-- ============================================================================

-- Function to update training session status
CREATE OR REPLACE FUNCTION update_training_session_status(
    session_uuid UUID,
    new_status VARCHAR(20),
    episode_count INTEGER DEFAULT NULL,
    reward_score DECIMAL(10,4) DEFAULT NULL
) RETURNS VOID AS $$
BEGIN
    UPDATE training_sessions 
    SET 
        status = new_status,
        total_episodes = COALESCE(episode_count, total_episodes),
        best_reward = GREATEST(COALESCE(reward_score, best_reward), best_reward),
        final_reward = COALESCE(reward_score, final_reward),
        updated_at = CURRENT_TIMESTAMP,
        completed_at = CASE WHEN new_status IN ('completed', 'failed', 'stopped') THEN CURRENT_TIMESTAMP ELSE completed_at END
    WHERE id = session_uuid;
END;
$$ LANGUAGE plpgsql;

-- Function to get live training metrics
CREATE OR REPLACE FUNCTION get_live_training_metrics(session_uuid UUID)
RETURNS TABLE(
    current_episode INTEGER,
    current_reward DECIMAL(10,4),
    average_reward DECIMAL(10,4),
    best_reward DECIMAL(10,4),
    policy_loss DECIMAL(10,6),
    value_loss DECIMAL(10,6),
    hoop_completion_rate DECIMAL(5,4),
    collision_rate DECIMAL(5,4)
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        MAX(tm.episode_number) as current_episode,
        (SELECT episode_reward FROM training_metrics WHERE session_id = session_uuid ORDER BY episode_number DESC LIMIT 1) as current_reward,
        AVG(tm.episode_reward)::DECIMAL(10,4) as average_reward,
        MAX(tm.episode_reward) as best_reward,
        (SELECT policy_loss FROM training_metrics WHERE session_id = session_uuid ORDER BY episode_number DESC LIMIT 1) as policy_loss,
        (SELECT value_loss FROM training_metrics WHERE session_id = session_uuid ORDER BY episode_number DESC LIMIT 1) as value_loss,
        AVG(tm.hoop_completion_rate)::DECIMAL(5,4) as hoop_completion_rate,
        AVG(tm.collision_rate)::DECIMAL(5,4) as collision_rate
    FROM training_metrics tm
    WHERE tm.session_id = session_uuid;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- TRIGGERS FOR AUTOMATIC UPDATES
-- ============================================================================

-- Trigger to update training session timestamp
CREATE OR REPLACE FUNCTION update_training_session_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_training_session_timestamp
    BEFORE UPDATE ON training_sessions
    FOR EACH ROW
    EXECUTE FUNCTION update_training_session_timestamp();

-- ============================================================================
-- DEFAULT HYPERPARAMETER CONFIGURATION
-- ============================================================================

-- Insert default hyperparameter configuration
INSERT INTO hyperparameter_configs (
    id, session_id, learning_rate, clip_ratio, entropy_coef, batch_size, 
    rollout_steps, num_epochs, gamma, gae_lambda, is_default
) VALUES (
    gen_random_uuid(), NULL, 0.0003, 0.2, 0.01, 64, 512, 10, 0.99, 0.95, TRUE
);

-- ============================================================================
-- COMMENTS FOR INTEGRATION
-- ============================================================================

COMMENT ON TABLE training_sessions IS 'Student training sessions with 10-180 minute time limits';
COMMENT ON TABLE hyperparameter_configs IS 'P3O hyperparameter configurations with student defaults';
COMMENT ON TABLE training_metrics IS 'Real-time training metrics from ClearML integration';
COMMENT ON TABLE hyperparameter_trials IS 'Random search hyperparameter optimization trials';
COMMENT ON TABLE model_checkpoints IS 'Saved model checkpoints (periodic/best/final)';
COMMENT ON TABLE optimization_suggestions IS 'AI-generated optimization suggestions';

COMMENT ON COLUMN training_sessions.training_minutes IS 'Student-required training duration (10-180 minutes, no default)';
COMMENT ON COLUMN training_sessions.user_id IS 'References your existing user table';
COMMENT ON COLUMN hyperparameter_configs.learning_rate IS 'P3O learning rate (1e-4 to 3e-3, default 3e-4)';
COMMENT ON COLUMN model_checkpoints.checkpoint_type IS 'periodic (every 10min), best (highest reward), final (end of training)';
COMMENT ON COLUMN hyperparameter_trials.trial_config IS 'JSON configuration for hyperparameter trial'; 