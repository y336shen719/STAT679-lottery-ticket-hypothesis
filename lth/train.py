import copy
import torch
from .pruning import apply_masks, mask_gradients


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += torch.nn.functional.cross_entropy(logits, y, reduction='sum').item()
        correct += (logits.argmax(1) == y).sum().item()
        total += len(y)
    return {'loss': loss_sum / total, 'accuracy': correct / total}


def train_fixed_steps(model, train_loader, val_loader, test_loader, masks, *, steps, lr, optimizer_name, eval_every, device,
                      progress=False, desc='train'):
    """`progress=True` shows a step-level tqdm bar with the latest validation metrics."""
    bar = None
    if progress:
        from tqdm import tqdm
        bar = tqdm(total=steps, desc=desc, leave=False, dynamic_ncols=True, position=1, mininterval=0.5)
    model.to(device)
    masks = {k: m.to(device) for k, m in masks.items()}  # copy once, not on every step
    apply_masks(model, masks)
    if optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError(optimizer_name)
    iterator = iter(train_loader)
    best_loss, best_step, best_state, history = float('inf'), 0, None, []
    for step in range(1, steps + 1):
        model.train()
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            x, y = next(iterator)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        mask_gradients(model, masks)
        optimizer.step()
        apply_masks(model, masks)
        if bar is not None:
            bar.update(1)
        if step % eval_every == 0 or step == steps:
            val = evaluate(model, val_loader, device)
            history.append({'step': step, 'val_loss': val['loss'], 'val_accuracy': val['accuracy']})
            if val['loss'] < best_loss:
                best_loss, best_step = val['loss'], step
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            if bar is not None:
                bar.set_postfix(val_loss=f"{val['loss']:.4f}", val_acc=f"{val['accuracy']:.4f}", best_step=best_step, refresh=False)
    if bar is not None:
        bar.close()
    end_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
    end_test = evaluate(model, test_loader, device)
    model.load_state_dict(best_state)
    best_test = evaluate(model, test_loader, device)
    model.load_state_dict(end_state)
    return {'best_step': best_step, 'best_val_loss': best_loss,
            'best_test_accuracy': best_test['accuracy'], 'end_test_accuracy': end_test['accuracy'],
            'history': history, 'end_state': end_state, 'best_state': best_state}
