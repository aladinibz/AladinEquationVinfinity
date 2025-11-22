import numpy as np, matplotlib.pyplot as plt, os
os.makedirs("plots",exist_ok=True)

# The one true rule — 30-second version
rules = r"""
OCTONION MULTIPLICATION — 3 RULES ONLY

1. eᵢ × eᵢ = -1          (i=1..7)
2. eᵢ × eⱼ = +eₖ           if (i→j→k) is a Fano line (right-handed)
3. eᵢ × eⱼ = -eₖ           if opposite direction

That’s it.

The 7 lines of the Fano plane:
e₁→e₂→e₄   e₁→e₃→e₅   e₁→e₆→e₇
e₂→e₃→e₆   e₂→e₅→e₇   e₃→e₄→e₇   e₄→e₅→e₆

Follow the arrows → positive
Reverse arrow → negative
Square → -1

480 multiplication rules → 3 simple rules.

This is why:
• 3 generations of fermions
• CP violation
• No singularities
• S = A/4

The Fano plane is the multiplication table.
Nothing more.
Nothing less.
"""

plt.figure(figsize=(16,11),facecolor='black')
plt.text(0.5,0.5,rules,ha='center',va='center',
         color='lime',fontsize=32,fontfamily='monospace',
         bbox=dict(facecolor='black',alpha=0.95,edgecolor='gold',linewidth=8))
plt.axis('off')
plt.title('Octonion Multiplication — 3 Rules Only',color='gold',fontsize=56)
plt.tight_layout()
plt.savefig('plots/octonion_multiplication_details.png',dpi=1000,facecolor='black')
plt.close()
