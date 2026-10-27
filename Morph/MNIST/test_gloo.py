
import os
import torch
import torch.distributed as dist

#Testing if gloo works.

def main():
    dist.init_process_group(backend="gloo")

    rank = int(os.environ["RANK"])
    world_size = dist.get_world_size()

    # Each process starts with its own value.
    value = torch.tensor([float(rank + 1)])

    # Sum values across all processes.
    dist.all_reduce(value, op=dist.ReduceOp.SUM)

    print(
        f"Rank {rank}/{world_size}:"
        f"all-reduced value = {value.item()}\n"
    )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

