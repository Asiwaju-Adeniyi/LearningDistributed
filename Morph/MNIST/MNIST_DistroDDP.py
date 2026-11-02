
import argparse
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, DistributedSampler
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import (
    ProfilerActivity,
    profile,
    record_function,
    schedule,
    tensorboard_trace_handler,
)


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


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "MNIST with manual gradient averaging and an optional "
            "PyTorch profiler run."
        )
    )
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=64)

    # Profiling is opt-in so the default training run is unchanged.
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Capture a short profiler trace for each rank and exit.",
    )
    # Use a separate directory from TorchDDP.py so the two scripts' traces
    # do not get mixed together in TensorBoard.
    parser.add_argument("--profile-dir", default="profiler_logs_manual")
    parser.add_argument("--profile-wait", type=int, default=1)
    parser.add_argument("--profile-warmup", type=int, default=2)
    parser.add_argument("--profile-active", type=int, default=3)

    args = parser.parse_args()

    if args.profile:
        if args.profile_wait < 0 or args.profile_warmup < 0:
            parser.error("--profile-wait and --profile-warmup must be >= 0")
        if args.profile_active < 1:
            parser.error("--profile-active must be >= 1")

    return args


def build_profiler(args, rank):
    # This Mac uses gloo/CPU distributed training. If the model is later
    # moved to CUDA, include the CUDA activity as well.
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    return profile(
        activities=activities,
        schedule=schedule(
            wait=args.profile_wait,
            warmup=args.profile_warmup,
            active=args.profile_active,
            repeat=1,
        ),
        # Each rank writes its own trace into the same directory. TensorBoard
        # groups these worker traces into one run and enables the Distributed
        # view when every worker contains communication events.
        on_trace_ready=tensorboard_trace_handler(
            args.profile_dir,
            worker_name=f"rank{rank}",
        ),
        record_shapes=True,
        with_stack=True,
    )


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
    args = parse_args()

    dist.init_process_group(backend="gloo")

    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        writer = None

        if rank == 0:
            writer = SummaryWriter("runs/mnist")

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
            batch_size=args.batch_size,
            sampler=sampler,
        )

        # 4. Create optimizer.
        optimizer = optim.Adadelta(model.parameters(), lr=1.0)

        model.train()

        # 5. Train. In profile mode, run only one epoch and stop after the
        # profiler's wait + warmup + active window.
        epochs = 1 if args.profile else args.epochs
        profile_steps = (
            args.profile_wait + args.profile_warmup + args.profile_active
        )
        if args.profile and profile_steps > len(loader):
            raise ValueError(
                f"The profiling window needs {profile_steps} batches, but "
                f"one epoch only has {len(loader)}."
            )
        profiler_ctx = (
            build_profiler(args, rank) if args.profile else nullcontext()
        )
        step = 0

        # Keep both ranks together before the profiling window starts.
        if args.profile:
            dist.barrier()

        with profiler_ctx as prof:
            for epoch in range(epochs):
                sampler.set_epoch(epoch)

                for batch_idx, (data, target) in enumerate(loader):
                    with record_function("train_step"):
                        optimizer.zero_grad()

                        with record_function("forward"):
                            output = model(data)

                        with record_function("loss"):
                            loss = F.nll_loss(output, target)

                        with record_function("backward"):
                            loss.backward()

                        # 6. Manually average gradients. This is one
                        # all-reduce per parameter, unlike DDP's buckets.
                        with record_function("manual_all_reduce"):
                            for param in model.parameters():
                                if param.grad is not None:
                                    dist.all_reduce(
                                        param.grad,
                                        op=dist.ReduceOp.SUM,
                                    )
                                    param.grad.div_(world_size)

                        # 7. Update parameters.
                        with record_function("optimizer_step"):
                            optimizer.step()

                    # Advance the profiler schedule once per training step.
                    if prof is not None:
                        prof.step()

                    step += 1

                    if batch_idx % 100 == 0:
                        print(
                            f"Rank {rank}, epoch {epoch + 1}, "
                            f"batch {batch_idx}, loss={loss.item():.4f}",
                            flush=True,
                        )

                    if args.profile and step >= profile_steps:
                        break

                if args.profile and step >= profile_steps:
                    break

        if args.profile:
            # Keep both ranks alive until every trace has been written.
            dist.barrier()

            print(
                f"\nRank {rank}: profiler trace written under "
                f"{args.profile_dir}",
                flush=True,
            )
            # Printing the full table from both ranks interleaves the output.
            # TensorBoard shows the per-rank comparison, so keep the console
            # table on rank 0 only.
            averages = prof.key_averages()
            if rank == 0:
                print("Rank 0 top ops:", flush=True)
                print(
                    averages.table(
                        sort_by="self_cpu_time_total",
                        row_limit=15,
                    ),
                    flush=True,
                )

            # This script does one all-reduce per parameter, so show the
            # communication calls explicitly.
            for event in averages:
                if event.key.startswith(("gloo:", "c10d::")):
                    print(
                        f"Rank {rank} {event.key}: calls={event.count}, "
                        f"self CPU={event.self_cpu_time_total / 1000.0:.2f} ms",
                        flush=True,
                    )

            if rank == 0:
                print(
                    f"\nOpen TensorBoard with: "
                    f"tensorboard --logdir {args.profile_dir}",
                    flush=True,
                )
        else:
            # 8. Verify parameters match across ranks.
            checkParamMatch(model, rank)

            print(f"Rank {rank} finished training.", flush=True)

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
