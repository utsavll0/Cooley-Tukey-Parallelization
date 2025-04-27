import matplotlib.pyplot as plt
import numpy as np

# Labels and sizes
size_labels = ['1024²', '2048²', '4096²', '8192²', '16384²']
x = np.arange(len(size_labels)) * 1.5  # label locations
width = 0.25  # bar width (still used for execution time)

# Timing data
data = {
    4: {'parallel': [271, 1248, 5156, 21185, 87964],
        'serial':   [549, 2488, 10879, 46692, 200445]},
    8: {'parallel': [248.306, 1058.1, 4173.24, 17165.4, 71813.1],
        'serial':   [665.575, 3048.05, 13367.6, 57568.8, 247993]},
    16:{'parallel': [203.778, 803.977, 3117.63, 12801.8, 52801.4],
        'serial':   [572.501, 2666.69, 11473, 49444.5, 211281]},
    32:{'parallel': [283.439, 724.593, 2717.01, 11123.6, 47924.9],
        'serial':   [569.731,2620.39,11250.4,49270.5,215701]}
}

# --- Figure 1: Execution Time (log scale, grouped bars) ---
plt.figure(figsize=(10, 6))
for i, n in enumerate([4, 8, 16, 32]):
    offset = (i - 1) * width
    plt.bar(x + offset, data[n]['parallel'], width, label=f'Parallel N={n}')
    plt.bar(x + offset, data[n]['serial'], width, bottom=data[n]['parallel'],
            label=f'Serial N={n}', alpha=0.3)

plt.ylabel('Execution Time (ms)')
plt.xlabel('Matrix Size')
plt.title('Execution Time (Log Scale)')
plt.xticks(x, size_labels)
plt.yscale('log')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('execution_time.png', dpi=300)
plt.show()

# --- Figure 2: Speedup (Line Plot) ---
plt.figure(figsize=(10, 6))
for n in [4, 8, 16, 32]:
    parallel = np.array(data[n]['parallel'])
    serial = np.array(data[n]['serial'])
    speedup = serial / parallel
    plt.plot(size_labels, speedup, marker='o', linewidth=2, label=f'N={n}')

plt.ylabel('Speedup (serial / parallel)')
plt.xlabel('Matrix Size')
plt.title('Speedup vs Matrix Size')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('speedup.png', dpi=300)
plt.show()