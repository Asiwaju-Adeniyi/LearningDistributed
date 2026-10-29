
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, DistributedSampler
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def checkParamMatch(model, rank):
    with torch.no_grad():
        all_match = True

        for param in model.parameters():
            ref = param.detach().clone()
            dist.broadcast(ref, src=0)

            if not torch.allclose(param, ref, atol=1e-6, rtol=1e-5):
                all_match = False

        print(
            f"Rank {rank}: parameters match rank 0 = {all_match}",
            flush=True,
        )


def main():
    dist.init_process_group(backend="gloo")

    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        model = Net()

        # 1. Synchronize initial parameters.
        with torch.no_grad():
            for param in model.parameters():
                dist.broadcast(param, src=0)

        # 2. Prepare MNIST.
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        dataset = datasets.MNIST(
            "../data",
            train=True,
            download=True,
            transform=transform,
        )

        # 3. Partition the dataset across ranks.
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )

        loader = DataLoader(
            dataset,
            batch_size=64,
            sampler=sampler,
        )

        # 4. Create optimizer.
        optimizer = optim.Adadelta(model.parameters(), lr=1.0)

        model.train()

        # 5. Train for 14 epochs.
        for epoch in range(14):
            sampler.set_epoch(epoch)

            for batch_idx, (data, target) in enumerate(loader):
                optimizer.zero_grad()

                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()

                # 6. Manually average gradients.
                for param in model.parameters():
                    if param.grad is not None:
                        dist.all_reduce(
                            param.grad,
                            op=dist.ReduceOp.SUM,
                        )
                        param.grad.div_(world_size)

                # 7. Update parameters.
                optimizer.step()

                if batch_idx % 100 == 0:
                    print(
                        f"Rank {rank}, epoch {epoch + 1}, "
                        f"batch {batch_idx}, loss={loss.item():.4f}",
                        flush=True,
                    )

        # 8. Verify parameters match across ranks.
        checkParamMatch(model, rank)

        print(f"Rank {rank} finished training.", flush=True)

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
