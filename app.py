import multiprocessing
import platform

print("CPU:", platform.processor())
print("Number of cores:", multiprocessing.cpu_count())