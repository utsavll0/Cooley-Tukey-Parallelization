import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Parse the data
data = '''Size,Serial,Parallel,Speedup
1024 ^ 2,120 ms,90 ms,1.28
2048 ^ 2,535 ms,387 ms,1.38
4096 ^ 2,2337 ms,1593 ms,1.46
8192 ^ 2,10009 ms,6530 ms,1.53
16384 ^ 2,42390 ms,26673 ms,1.58'''

# Convert string data to DataFrame
lines = data.strip().split('\n')
headers = lines[0].split(',')
rows = []
for line in lines[1:]:
    values = line.split(',')
    size = int(values[0].split('^')[0].strip())
    serial_time = int(values[1].split()[0])
    parallel_time = int(values[2].split()[0])
    speedup = float(values[3])
    rows.append([size, serial_time, parallel_time, speedup])

df = pd.DataFrame(rows, columns=['Size', 'Serial', 'Parallel', 'Speedup'])

# Create figure with two subplots

plt.figure(figsize=(10, 6))

# Plot 1: Execution times (Serial vs Parallel)
# Create a bar graph for execution times
x = np.arange(len(df['Size']))  # Create x positions for the bars
width = 0.35  # Width of the bars

plt.bar(x - width/2, df['Serial'], width, label='Serial', color='#0FA3B1')
plt.bar(x + width/2, df['Parallel'], width, label='Parallel', color='#EDDEA4')

# Add data labels on top of each bar
for i, value in enumerate(df['Serial']):
    plt.text(x[i] - width/2, value, f"{value} ms", ha='center', va='bottom')
for i, value in enumerate(df['Parallel']):
    plt.text(x[i] + width/2, value, f"{value} ms", ha='center', va='bottom')

# Set the x-axis ticks and labels
plt.yscale('log')  # Set y-axis to log scale
plt.xticks(x, [f"{size}^2" for size in df['Size']])
plt.xlabel('Input Size')
plt.ylabel('Execution Time (ms)')
plt.title('Serial vs Parallel Execution Time')
# plt.grid(True, which="both", ls="--", alpha=0.7, axis='y')
plt.legend()
plt.tight_layout()
plt.show()

# Add data labels to the first plot
plt.figure(figsize=(10, 6))
for i, row in df.iterrows():
    plt.annotate(f"{row['Serial']} ms", 
                 (row['Size'], row['Serial']), 
                 textcoords="offset points", 
                 xytext=(0,10), 
                 ha='center')
    plt.annotate(f"{row['Parallel']} ms", 
                 (row['Size'], row['Parallel']), 
                 textcoords="offset points", 
                 xytext=(0,-15), 
                 ha='center')

# Plot 2: Speedup
plt.plot(df['Size'], df['Speedup'], 'o-', color='green')
plt.xlabel('Input Size')
plt.ylabel('Speedup (Serial/Parallel)')
plt.title('Speedup vs Matrix Size')
plt.xscale('log', base=2)
# plt.grid(True, which="both", ls="--", alpha=0.7)

# Add data labels to the second plot
for i, row in df.iterrows():
    plt.annotate(f"{row['Speedup']:.2f}x", 
                 (row['Size'], row['Speedup']), 
                 textcoords="offset points", 
                 xytext=(0,10), 
                 ha='center')

# Set a tight layout to prevent overlap
plt.tight_layout()

# Show the plots
plt.show()