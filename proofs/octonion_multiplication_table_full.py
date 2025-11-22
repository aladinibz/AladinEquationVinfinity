import numpy as np, matplotlib.pyplot as plt, os
os.makedirs("plots",exist_ok=True)

# The one true table — Fano plane in matrix form
labels = ['1', 'e₁', 'e₂', 'e₃', 'e₄', 'e₅', 'e₆', 'e₇']
table = np.array([
    [ 1,  0,  0,  0,  0,  0,  0,  0],  # 1 × anything = itself
    [ 0, -1,  4,  5,  7, -6, -3,  2],  # e₁ row
    [ 0, -4, -1,  6, -5,  7,  1, -3],  # e₂
    [ 0, -5, -6, -1,  6, -4,  2,  1],  # e₃
    [ 0, -7,  5, -6, -1,  3, -2,  1],  # e₄
    [ 0,  6, -7,  4, -3, -1,  1, -2],  # e₅
    [ 0,  3, -1, -2,  2, -1, -1,  7],  # e₆
    [ 0, -2,  3, -1, -1,  2, -7, -1]   # e₇
])

# Convert to signed e_k notation (0=1, positive=e_k, negative=-e_|k|)
def format_entry(n):
    if n == 0: return '1'
    if n == -1: return '-1'
    sign = '+' if n > 0 else '-'
    k = abs(n)
    return f'{sign}e{k}' if k > 0 else '-1'

# Plot the sacred table
plt.figure(figsize=(20,16),facecolor='black')
plt.imshow(np.abs(table), cmap='plasma', vmin=0, vmax=7)
for i in range(8):
    for j in range(8):
        val = table[i,j]
        txt = format_entry(val)
        color = 'lime' if abs(val) in [1,4,5,6,7] else 'white'
        plt.text(j,i,txt,ha='center',va='center',color=color,fontsize=28,fontweight='bold')

plt.xticks(range(8),labels,color='white',fontsize=32)
plt.yticks(range(8),labels,color='white',fontsize=32)
plt.title('Octonion Multiplication Table — The True DNA of Reality',color='gold',fontsize=52,pad=60)
plt.text(4,-0.8,'7 imaginary units → 7 forces of creation\n'
                 'Non-associative → quantum uncertainty\n'
                 'Alternative → no singularities\n'
                 '480 multiplication rules → all of physics',
         ha='center',color='cyan',fontsize=34,bbox=dict(facecolor='black',alpha=0.95))

plt.tight_layout()
plt.savefig('plots/octonion_multiplication_table_full.png',dpi=1200,facecolor='black')
plt.close()
