import time
import random
import tracemalloc
from numba import njit

@njit(fastmath=True)
def quicksort(arr):
    stack = [(0, len(arr) - 1)]
    while stack:
        low, high = stack.pop()
        while low < high:
            p = partition(arr, low, high)
            if p - low < high - p:
                stack.append((p + 1, high))
                high = p
            else:
                stack.append((low, p))
                low = p + 1

@njit(fastmath=True, inline='always')
def partition(arr, low, high):
    mid = (low + high) // 2
    pivot = median_of_three(arr, low, mid, high)
    i = low
    j = high
    while True:
        while arr[i] < pivot:
            i += 1
        while arr[j] > pivot:
            j -= 1
        if i >= j:
            return j
        arr[i], arr[j] = arr[j], arr[i]
        i += 1
        j -= 1

@njit(fastmath=True, inline='always')
def median_of_three(arr, low, mid, high):
    if arr[low] > arr[mid]:
        arr[low], arr[mid] = arr[mid], arr[low]
    if arr[low] > arr[high]:
        arr[low], arr[high] = arr[high], arr[low]
    if arr[mid] > arr[high]:
        arr[mid], arr[high] = arr[high], arr[mid]
    return arr[mid]

def measure_performance(data):
    tracemalloc.start()
    start_time = time.perf_counter()

    data_copy = data.copy()
    quicksort(data_copy)

    end_time = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()

    tracemalloc.stop()
    execution_time = (end_time - start_time) * 1000  # Convertir a milisegundos

    return execution_time, peak / 1024 / 1024  # Devuelve el tiempo de ejecución y uso de memoria en MiB

# Generar datos de prueba
small_data = [random.randint(0, 1000) for _ in range(100)]
medium_data = [random.randint(0, 1000) for _ in range(300)]
large_data = [random.randint(0, 1000) for _ in range(500)]

# Realizar llamadas de "calentamiento"
warmup_data = [random.randint(0, 1000) for _ in range(100)]
quicksort(warmup_data)

# Medir el rendimiento para cada conjunto de datos
execution_time_small, memory_usage_small = measure_performance(small_data)
execution_time_medium, memory_usage_medium = measure_performance(medium_data)
execution_time_large, memory_usage_large = measure_performance(large_data)

# Imprimir los resultados de rendimiento
print("Conjunto Pequeño (100 elementos):")
print(f"Tiempo de Ejecución: {execution_time_small:.2f} ms")
print(f"Uso de Memoria: {memory_usage_small:.2f} MiB")

print("\nConjunto Mediano (300 elementos):")
print(f"Tiempo de Ejecución: {execution_time_medium:.2f} ms")
print(f"Uso de Memoria: {memory_usage_medium:.2f} MiB")

print("\nConjunto Grande (500 elementos):")
print(f"Tiempo de Ejecución: {execution_time_large:.2f} ms")
print(f"Uso de Memoria: {memory_usage_large:.2f} MiB")
