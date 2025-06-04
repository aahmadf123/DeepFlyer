import torch
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
import numpy as np
from pathlib import Path
from PIL import Image
from stable_baselines3.common.callbacks import BaseCallback

class GradCAMXAI:
    """
    Grad-CAM explainer for convolutional policies.
    """
    def __init__(self, model, target_layer: str):
        """
        Args:
            model: the RL model (e.g., PPO or SAC)
            target_layer: dot-separated path to target module in policy, e.g.
                          'policy.features_extractor.cnn.4'
        """
        # Traverse to get the target module
        module = model
        for attr in target_layer.split('.'):
            module = getattr(module, attr)
        self.model = model
        self.cam_extractor = GradCAM(model=model.policy, target_layer=module)

    def __call__(self, input_tensor: torch.Tensor) -> np.ndarray:
        """
        Compute saliency map for the given input tensor.
        input_tensor: FloatTensor of shape [1,C,H,W]
        Returns: 2D numpy array of saliency values.
        """
        # Ensure eval mode
        self.model.policy.eval()
        # Forward pass
        with torch.no_grad():
            out = self.model.policy(input_tensor)
        # Choose target class/logit
        if hasattr(out, 'argmax'):
            target = out.argmax(dim=1).item()
        else:
            target = 0
        # Generate CAM
        activation_map = self.cam_extractor(target, out)
        sal = activation_map[0]
        # Convert to numpy
        return sal

class XAISaliencyCallback(BaseCallback):
    """Callback that periodically captures saliency maps and saves images."""
    def __init__(self, xai_extractor: GradCAMXAI, sal_dir: Path, interval: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.xai = xai_extractor
        self.sal_dir = Path(sal_dir)
        self.interval = interval
        self.counter = 0
        self.sal_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        self.counter += 1
        if self.counter % self.interval == 0:
            env = self.model.env
            obs = getattr(env, 'last_obs', None)
            if obs is None:
                return True
            # Assume obs is image HxWxC, convert to CHW
            img = obs
            if isinstance(img, np.ndarray) and img.ndim == 3:
                # convert to tensor
                inp = torch.tensor(img, dtype=torch.float32).permute(2,0,1).unsqueeze(0)
            else:
                return True
            sal = self.xai(inp)
            # Normalize saliency map
            sal_norm = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
            sal_img = Image.fromarray((sal_norm * 255).astype('uint8')).convert('L')
            sal_img.save(str(self.sal_dir / f"saliency_{self.counter}.png"))
        return True 