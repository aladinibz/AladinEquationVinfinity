import numpy as np, matplotlib.pyplot as plt, os
os.makedirs("plots",exist_ok=True)

# Fano plane points: 7 points + 1 center
points = {
    '1': (0, 0),      # center
    'e₁': (0, 1),      # top
    'e₂': (0.866, -0.5), # bottom right
    'e₃': (-0.866, -0.5), # bottom left
    'e₄': (0.433, 0.25),  # middle right
    'e₅': (-0.433, 0.25), # middle left
    'e₆': (0, -0.5),      # bottom
    'e₇': (0, -1)         # very bottom (circle)
}

# The 7 sacred lines (each line = multiplication rule)
lines = [
    ('e₁','e₂','e₄'), ('e₁','e₃','e₅'), ('e₁','e₆','e₇'),
    ('e₂','e₃','e₆'), ('e₂','e₅','e₇'), ('e₃','e₄','e₇'),
    ('e₄','e₅','e₆')
]

plt.figure(figsize=(12,13),facecolor='black')

# Draw the circle (e₇ path)
circle = plt.Circle((0,0),1,color='lime',fill=False,lw=6)
plt.gca().add_artist(circle)

# Draw the 7 lines
for a,b,c in lines:
    ax,ay = points[a]; bx,by = points[b]; cx,cy = points[c]
    plt.plot([ax,bx,cx,ax],[ay,by,cy,ay],color='gold',lw=5)

# Plot points
for label,(x,y) in points.items():
    color = 'white' if label=='1' else 'lime'
    size = 800 if label=='1' else 1200
    plt.scatter(x,y,s=size,color=color,edgecolor='gold',linewidth=3,zorder=5)
    plt.text(x,y+0.15,label,color=color,ha='center',fontsize=28,fontweight='bold')

plt.xlim(-1.3,1.3); plt.ylim(-1.4,1.4)
plt.axis('off')
plt.title('Fano Plane — Geometric Multiplication Table of Octonions',color='gold',fontsize=32,pad=40)
plt.text(0,-1.6,'Each line = e_i × e_j = e_k\n'
                 '7 points = 7 imaginary units\n'
                 'Center = real unit 1\n'
                 'Circle = e₇ cycle',
         ha='center',color='cyan',fontsize=24,bbox=dict(facecolor='black',alpha=0.9))

plt.tight_layout()
plt.savefig('plots/fano_plane_geometric.png',dpi=700,facecolor='black')
plt.close()
