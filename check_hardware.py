import psutils
import torch
def check_hardware_requirements():
    # Check RAM
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)  # Convert bytes to GB
    print(f"Total RAM: {ram_gb:.2f} GB")
    if ram_gb < MIN_RAM_GB:
        print(f"Insufficient RAM. Minimum required is {MIN_RAM_GB} GB.")
    else:
        print("RAM requirement met.")
    
    # Check CPU cores
    cpu_cores = psutil.cpu_count(logical=False)  # Get physical cores only
    print(f"Total CPU cores: {cpu_cores}")
    if cpu_cores < MIN_CPU_CORES:
        print(f"Insufficient CPU cores. Minimum required is {MIN_CPU_CORES}.")
    else:
        print("CPU core requirement met.")
    
    # Overall check
    if ram_gb >= MIN_RAM_GB and cpu_cores >= MIN_CPU_CORES:
        return True
    else:
        return False
    
