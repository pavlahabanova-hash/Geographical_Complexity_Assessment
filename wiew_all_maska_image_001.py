import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# --- NASTAVENÍ ---
PATH = r'C:\user\PAVLA\dataset'
N = 50  # Rozměr mřížky (50x50)
TILE_SIZE = 256

def interaktivni_kontrola(base_path, n, tile_size):
    img_dir = os.path.join(base_path, 'images')
    mask_dir = os.path.join(base_path, 'masks')
    
    velky_rozmer = n * tile_size
    # Vytvoření velkých polí (array) pro rychlé vykreslení v Matplotlibu
    velka_foto = np.zeros((velky_rozmer, velky_rozmer, 3), dtype=np.uint8)
    velka_maska = np.zeros((velky_rozmer, velky_rozmer), dtype=np.uint8)

    print("Skládám mapy do paměti pro interaktivní prohlížení...")

    for i in range(n):
        for j in range(n):
            filename = f"tile_{i:02d}_{j:02d}.png"
            # Výpočet pozice (pozor na orientaci os)
            px = i * tile_size
            py = (n - 1 - j) * tile_size 
            
            p_img = os.path.join(img_dir, filename)
            p_mask = os.path.join(mask_dir, filename)

            if os.path.exists(p_img):
                tile = Image.open(p_img).convert('RGB')
                velka_foto[py:py+tile_size, px:px+tile_size] = np.array(tile)
                
            if os.path.exists(p_mask):
                mask_tile = Image.open(p_mask).convert('L')
                velka_maska[py:py+tile_size, px:px+tile_size] = np.array(mask_tile)

    # --- VYTVOŘENÍ PROPOJENÉHO ZOBRAZENÍ ---
    # sharex a sharey zajistí, že zoomování na jednom grafu ovládá i druhý
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)
    
    ax1.imshow(velka_foto)
    ax1.set_title("Fotomapa (Zoomuj zde!)")
    
    ax2.imshow(velka_maska, cmap='gray')
    ax2.set_title("Maska (Pohybuje se automaticky)")

    # Odstranění bílých okrajů
    plt.tight_layout()
    
    print("\nNÁVOD:")
    print("1. Vyber nástroj 'Lupa' (Zoom) v dolní liště okna.")
    print("2. Táhni myší v jednom z oken pro přiblížení.")
    print("3. Druhé okno se přiblíží na úplně stejné místo.")
    
    plt.show()

if __name__ == "__main__":
    interaktivni_kontrola(PATH, N, TILE_SIZE)# micromamba activate pavla