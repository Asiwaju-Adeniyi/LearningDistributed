
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, DistributedSampler
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP


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


def evaluate(model, test_loader):
    model.eval()

    stats = torch.zeros(2, dtype=torch.int64)

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            predictions = output.argmax(dim=1)

            stats[0] += (predictions == target).sum().item()
            stats[1] += target.size(0)

    dist.all_reduce(stats, op=dist.ReduceOp.SUM)

    accuracy = 100.0 * stats[0].item() / stats[1].item()

    if dist.get_rank() == 0:
        print(f"Test accuracy: {accuracy:.2f}%", flush=True)

    model.train()


def main():
    dist.init_process_group(backend="gloo")

    try:
        # 1. Create the model and wrap it with DDP.
        model = DDP(Net())

        optimizer = optim.Adadelta(model.parameters(), lr=1.0)

        # 2. Prepare the datasets.
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_dataset = datasets.MNIST(
            "../data",
            train=True,
            download=True,
            transform=transform,
        )

        test_dataset = datasets.MNIST(
            "../data",
            train=False,
            download=True,
            transform=transform,
        )

        # 3. Partition the data automatically by rank.
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        test_sampler = DistributedSampler(test_dataset, shuffle=False)

        train_loader = DataLoader(
            train_dataset,
            batch_size=64,
            sampler=train_sampler,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=64,
            sampler=test_sampler,
        )

        # 4. Train.
        for epoch in range(14):
            model.train()
            train_sampler.set_epoch(epoch)

            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()

                output = model(data)
                loss = F.nll_loss(output, target)

                loss.backward()  # DDP synchronizes gradients.
                optimizer.step()

                if batch_idx % 100 == 0 and dist.get_rank() == 0:
                    print(
                        f"Epoch {epoch + 1}, batch {batch_idx}, "
                        f"loss={loss.item():.4f}",
                        flush=True,
                    )

            # 5. Evaluate after each epoch.
            evaluate(model, test_loader)

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
