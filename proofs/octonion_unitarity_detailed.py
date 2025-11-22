import numpy as np, matplotlib.pyplot as plt, os
os.makedirs("plots",exist_ok=True)

text = r"""
OCTONION UNITARITY — WHY INFORMATION IS ETERNAL

1. Octonion O = x₀ + x₁e₁ + ... + x₇e₇ ∈ 𝕆
   Norm: |O|² = x₀² + x₁² + ... + x₇² = Tr[O† O]

2. From ℒ₆: λ → +∞ enforces
   Tr[O† O] = 1   ∀ x ∈ ℝ^{1,3}

   → Every point in spacetime is a unit octonion
   → |O(x)| = 1 exactly

3. Evolution equation from ℒ₅ + ℒ₆:
   i ∂_t O = H O      with    H† = H
   → Hamiltonian Hermitian → unitary evolution

4. Multiplication table preserves norm:
   |O₁ O₂| = |O₁| |O₂| = 1 × 1 = 1

   → Even non-associative multiplication keeps norm = 1

5. Physical consequences:
   • No information loss — ever
   • Black hole evaporation: pure → mixed → pure again
   • Quantum gravity finite (no UV divergences)
   • Consciousness = coherent octonion state

6. Mathematical proof:
   d/dt |O|² = (∂_t O†) O + O† (∂_t O) 
            = −i H† O† O + i O† H O = 0

   → |O|² conserved exactly → unitarity

The universe cannot lose information
because every point in spacetime has norm 1
and evolves unitarily on S⁷.

This is the deepest reason
why physics works.
"""

plt.figure(figsize=(22,18),facecolor='black')
plt.text(0.5,0.5,text,ha='center',va='center',
         color='lime',fontsize=22,fontfamily='monospace',
         bbox=dict(facecolor='black',alpha=0.95,edgecolor='gold',linewidth=8))
plt.axis('off')
plt.title('Octonion Unitarity — Information Is Eternal',color='gold',fontsize=56,pad=70)
plt.text(0.5,0.02,"Tr[O† O] = 1 → The Universe Remembers Everything",
         ha='center',color='cyan',fontsize=42)
plt.tight_layout()
plt.savefig('plots/octonion_unitarity_detailed.png',dpi=1200,facecolor='black')
plt.close()
