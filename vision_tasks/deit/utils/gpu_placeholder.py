import torch
import time
import os

def gpu_burn(gpu_id, matrix_size=1024, sleep_interval=0.01):
    """在指定 GPU 上持续做矩阵乘法，占据一定利用率"""
    device = torch.device(f"cuda:{gpu_id}")
    a = torch.randn(matrix_size, matrix_size, device=device)
    b = torch.randn(matrix_size, matrix_size, device=device)
    while True:
        _ = torch.mm(a, b)
        time.sleep(sleep_interval)

if __name__ == "__main__":
    import multiprocessing
    num_gpus = int(os.environ.get("NUM_GPUS", 4))
    matrix_size = int(os.environ.get("BURN_MATRIX_SIZE", 4096))
    sleep_interval = float(os.environ.get("BURN_SLEEP", 0.05))

    processes = []
    for i in range(num_gpus):
        p = multiprocessing.Process(target=gpu_burn, args=(i, matrix_size, sleep_interval))
        p.daemon = True
        p.start()
        processes.append(p)

    # 主进程永远等待，直到被 kill
    for p in processes:
        p.join()