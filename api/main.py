from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, File, Depends
from api.dependencies import get_reward_registry, get_algorithms, get_status_dir
from api.schemas import TrainRequest
from api.workers import start_training_worker
from pathlib import Path
import json
import uuid
from rl_agent.sandbox import RewardSandbox

app = FastAPI(
    title="DeepFlyer RL API",
    version="0.1.0"
)

# status files are created in working directory by default

@app.get("/api/rewards/list")
def list_rewards(registry=Depends(get_reward_registry)):
    """
    Return the list of registered reward presets (id and label).
    """
    return registry.list_presets()

@app.get("/api/algorithms/list")
def list_algorithms(algos=Depends(get_algorithms)):
    """Return list of supported algorithms for Explorer and Researcher modes."""
    return algos

@app.post("/api/train/start")
def train_start(req: TrainRequest, background_tasks: BackgroundTasks):
    """Launch a training job in the background."""
    cfg = req.dict()
    job_id = start_training_worker(cfg)
    # Possibly push background_tasks.add_task if we had async; but subprocess already detached.
    return {"job_id": job_id, "status": "starting"}

@app.get("/api/train/status/{job_id}")
def train_status(job_id: str, status_dir=Depends(get_status_dir)):
    """Return current status for a job."""
    status_file = status_dir / f"status_{job_id}.json"
    if not status_file.exists():
        raise HTTPException(status_code=404, detail="Job ID not found")
    with status_file.open() as f:
        return json.load(f)

@app.post("/api/reward/validate")
def reward_validate(file: UploadFile = File(...)):
    """Validate uploaded Python file has custom_reward function with correct signature."""
    tmp_path = Path(f"/tmp/{uuid.uuid4()}_{file.filename}")
    content = file.file.read()
    tmp_path.write_bytes(content)
    try:
        sandbox = RewardSandbox(tmp_path)
        ok, msg = sandbox.test_dummy()
        status = "pass" if ok else "fail"
    except Exception as e:
        status = "error"
        msg = str(e)
    finally:
        tmp_path.unlink(missing_ok=True)
    return {"status": status, "message": msg}
