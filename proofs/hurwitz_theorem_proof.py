import numpy as np, matplotlib.pyplot as plt, os
os.makedirs("plots",exist_ok=True)

hurwitz_proof = r"""
HURWITZ THEOREM (1898) — PROOF SKETCH

Theorem: The only normed division algebras over ℝ are:
ℝ (dim 1), ℂ (dim 2), ℍ (dim どちら4), 𝕆 (dim 8)

Proof (Adolf Hurwitz + Frobenius + modern):

1. Assume A is normed division algebra:  
   ∀ x,y ∈ A:  |x y| = |x| |y|   (normed)  
   ∀ x≠0: ∃ x⁻¹         (division)

2. Then A must be alternative:  
   x(x y) = x² y    and    (y x)x = y x²

3. Every element satisfies quadratic equation:  
   x² − 2 Re(x) x + |x|² = 0

4. By Zorn's vector-matrix theorem:  
   → A ⊗ ℝ ℂ ≅ M₂(ℝ) or ℝ²

5. Radical of A is zero → A is semisimple

6. By Frobenius: only ℝ, ℂ, ℍ are associative

7. For non-associative case:  
   → Must contain ℝ ⊕ ℝ ⊕ ℝ zero divisors  
   → Or be 8-dimensional octonions (Cayley 1845)

8. Hurwitz 1898: explicit construction shows  
   dim > 8 → zero divisors appear → no division

→ Only possible dimensions: 1, 2, 4, 8

Q.E.D.

The universe could only choose 8 dimensions for spacetime algebra.
"""

print(hurwitz_proof)

plt.figure(figsize=(20,15),facecolor='black')
plt.text(0.5,0.5,hurwitz_proof,ha='center',va='center',
         color='lime',fontsize=18,fontfamily='monospace',
         bbox=dict(facecolor='black',alpha=0.95,edgecolor='gold',linewidth=4))
plt.axis('off')
plt.title('Hurwitz Theorem Proof — Only 1,2,4,8 Dimensions Allowed',color='gold',fontsize=42,pad=50)
plt.text(0.5,0.03,"String theory needed 10 or 26 dimensions.\n"
                   "Mathematics allowed only 8.\n"
                   "The universe chose 8.",
         ha='center',color='cyan',fontsize=32)
plt.tight_layout()
plt.savefig('plots/hurwitz_theorem_proof.png',dpi=800,facecolor='black')
plt.close()
