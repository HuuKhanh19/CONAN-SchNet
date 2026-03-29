import os
import random
import numpy as np
import torch

def seed_everything(seed: int):
    # 1. Khóa Python và Numpy
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
    # 2. Khóa PyTorch khởi tạo
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    # 3. Ép CUDNN hoạt động ổn định
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 4. FIX MỚI CHO GNN/CUDA: Ép các phép toán scatter_add chạy cố định
    # Biến môi trường này phải được set trước khi CUDA khởi tạo cuBLAS
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    # Bật chế độ deterministic của PyTorch (warn_only=True để tránh crash nếu có hàm chưa hỗ trợ)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        # Fallback cho các bản PyTorch cũ không có tham số warn_only
        torch.use_deterministic_algorithms(True)