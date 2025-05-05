import torch
from transformers import TrainerCallback
from transformers import Trainer


class TracinTrainer(Trainer):
    def training_step(self, model, inputs, optimizer=None):
        # Save the current batch inputs to all callbacks
        for callback in self.callback_handler.callbacks:
            if hasattr(callback, "set_current_inputs"):
                callback.set_current_inputs(inputs)

        return super().training_step(model, inputs)


class TracinCallback(TrainerCallback):
    def __init__(self, capture_every_n_steps=1):
        super().__init__()
        self.checkpoint_grads = []
        self.capture_every_n_steps = capture_every_n_steps
        self.inputs = None

    def set_current_inputs(self, inputs):
        self.inputs = inputs

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.capture_every_n_steps != 0:
            return

        model = kwargs["model"]

        if self.inputs is None:
            print(f"Step {state.global_step}: No inputs found.")
            return

        print(f"Capturing gradients at step {state.global_step}")
        model.zero_grad()
        outputs = model(**self.inputs)
        loss = outputs.loss
        loss.backward()

        grads = torch.cat(
            [p.grad.view(-1) for p in model.parameters() if p.grad is not None]
        )
        self.checkpoint_grads.append(grads.detach().cpu())

    def on_train_end(self, args, state, control, **kwargs):
        # Optionally: log or process the gradients once training finishes
        pass


# Compute influence scores
def compute_influence(grads_checkpoints, grads_target):
    influence_scores = []
    for ckpt_grad in grads_checkpoints:
        influence_score = torch.dot(grads_target.to(ckpt_grad.device), ckpt_grad)
        influence_scores.append(influence_score.item())
    return influence_scores


def get_gradient_for_input(model, input_batch, device):
    """
    Get the gradient for a single input batch.
    """
    outputs = model(**input_batch)
    loss = outputs.loss
    loss.backward()

    grads = torch.cat(
        [p.grad.view(-1) for p in model.parameters() if p.grad is not None]
    )
    # print(grads)

    model.zero_grad()
    return grads


# checkpoint_grads = torch.load("checkpoint_grads.pt")
# target_grad = get_gradient_for_input(model, one_batch_from_dataset, device)

# scores = compute_influence(checkpoint_grads, target_grad)
