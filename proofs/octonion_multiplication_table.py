import numpy as np, matplotlib.pyplot as plt, os
os.makedirs("plots",exist_ok=True)

# THE FANO PLANE — OCTONION MULTIPLICATION RULES (standard)
# e_i × e_j = -δ_{ij} e_0 + ε_{ijk} e_k   (e_0 = 1)

labels = ['1', 'e₁', 'e₂', 'e₃', 'e₄', 'e₅', 'e₆', 'e₇']
table = np.zeros((8,8), dtype=int)

# Diagonal: e_i × e_i = -1
for i in range(1,8):
    table[i,i] = -1

# Fano plane cycles (oriented)
cycles = [
    (1,2,4), (1,3,5), (1,6,7),
    (2,3,6), (2,5,7), (3,4,7),
    (4,5,6)
]

for a,b,c in cycles:
    table[a,b], table[b,c], table[c,a] = c, a, b      # positive orientation
    table[b,a], table[c,b], table[a,c] = -c, -a, -b   # negative

# Plot the sacred table
plt.figure(figsize=(14,12),facecolor='black')
plt.imshow(table, cmap='plasma', vmin=-7, vmax=7)
plt.colorbar(shrink=0.8, label='Product e_i × e_j → e_k  (sign+)')
for i in range(8):
    for j in range(8):
        val = table[i,j]
        color = 'white' if abs(val)>3 else 'lime'
        plt.text(j,i,f'{val:+d}',ha='center',va='center',
                 color=color,fontsize=20,fontweight='bold')

plt.xticks(range(8),labels,color='white',fontsize=18)
plt.yticks(range(8),labels,color='white',fontsize=18)
plt.title('Octonion Multiplication Table — The DNA of Reality',color='gold',fontsize=34)

plt.text(4,-0.8,'7 imaginary units → 7 generations of matter\n'
                 'Non-associative → quantum uncertainty\n'
                 'Alternative → no division by zero → no singularities',
         ha='center',color='cyan',fontsize=24,bbox=dict(facecolor='black',alpha=0.9))

plt.tight_layout()
plt.savefig('plots/octonion_multiplication_table.png',dpi=700,facecolor='black')
plt.close()
