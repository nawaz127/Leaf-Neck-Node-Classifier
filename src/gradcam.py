import torch
import torch.nn.functional as F

class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: str):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None

        layer = self._get_module(target_layer)
        layer.register_forward_hook(self._forward_hook)
        layer.register_full_backward_hook(self._backward_hook)

    def _get_module(self, dotted_name: str):
        module = self.model
        for attr in dotted_name.split('.'):
            module = module[int(attr)] if attr.isdigit() else getattr(module, attr)
        return module

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    @torch.no_grad()
    def _normalize(self, cam: torch.Tensor) -> torch.Tensor:
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> torch.Tensor:
        logits = self.model(input_tensor)
        self.model.zero_grad()
        grad_target = torch.zeros_like(logits)
        grad_target[0, class_idx] = 1.0
        logits.backward(gradient=grad_target)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = self._normalize(cam[0, 0])
        return cam
