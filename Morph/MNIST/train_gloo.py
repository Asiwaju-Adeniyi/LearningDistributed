
import os
import torch
import torch.distributed as dist

# Learning broadcast, all reduce, gather, send and recieve, etc. 
# Basically Communications Primitives in Distributed using PyTorch Distributed. 


def main():
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    broadcast()
    allReduce(rank, world_size)
    sendRecieve()

    dist.destroy_process_group()


##Broadcast Helper
def broadcast(): 

    # Each process starts with its own value.
    value = torch.randn(2,3)
    print(value)

    dist.broadcast(value, src = 0)
    print(f"Broadcasted Tensor: {value}")


##All Reduce Helper 
def allReduce(rank, world_size): 

    # Each process starts with its own value.
    value = torch.randn(2,3)
    print(value)

    # Sum values across all processes.
    dist.all_reduce(value, op=dist.ReduceOp.SUM)

    print(
        f"Rank {rank}/{world_size}:"
        f"all-reduced value = {value}\n")

    

## Send and Recieve Helper 
def sendRecieve(): 
    #Process Group is initialized. This sends a tensor from one GPU (rank) to another GPU (rank).
    rank = dist.get_rank() #This gets and stores each 

    if rank == 1: #the second GPU sends its data (tensor here) to the fourth GPU). These conditional statments masks out other GPUs. 
        tensor = torch.randn(2,3)
        dist.send(tensor, dst=3)
        print(f"Sent: {tensor}")

    elif rank == 3: #the fourth GPU receives data from the second GPU (tensor here). These conditional statements masks out other GPUs in the training fleet. 
        tensor = torch.zeros(2,3)
        dist.recv(tensor, src=1)
        print(f"Recieved: {tensor}")


if __name__ == "__main__":
    main()