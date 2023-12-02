import torch
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path, device='cuda'):
    model.load_state_dict(torch.load(path))
    model.to(device)
    return model
