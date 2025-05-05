import torch
from transformers import TrainerCallback


class TracinCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.checkpoint_grads = []
        self.last_inputs = None

    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs["model"]
        inputs = kwargs.get("inputs")  # Capture inputs on every step
        if inputs is None:
            return  # Skip if inputs are not available

        # Perform gradient calculation only if inputs are present
        model.zero_grad()
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()

        grads = torch.cat(
            [p.grad.view(-1) for p in model.parameters() if p.grad is not None]
        )
        self.checkpoint_grads.append(grads.detach().cpu())  # Store gradient on CPU

    def on_train_end(self, args, state, control, **kwargs):
        # Optionally: log or process the gradients once training finishes
        pass


# Compute influence scores
def compute_influence(grads_checkpoints, grads_target):
    influence_scores = []
    for ckpt_grad in grads_checkpoints:
        influence_score = torch.dot(grads_target, ckpt_grad)
        influence_scores.append(influence_score.item())
    return influence_scores


def get_gradient_for_input(model, input_batch, device):
    """
    Get the gradient for a single input batch.
    """
    model.zero_grad()
    outputs = model(**input_batch)
    loss = outputs.loss
    loss.backward()

    grads = torch.cat(
        [p.grad.view(-1) for p in model.parameters() if p.grad is not None]
    )

    model.zero_grad()
    return grads


# checkpoint_grads = torch.load("checkpoint_grads.pt")
# target_grad = get_gradient_for_input(model, one_batch_from_dataset, device)

# scores = compute_influence(checkpoint_grads, target_grad)
