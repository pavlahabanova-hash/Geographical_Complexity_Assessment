import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# --- NASTAVENÍ ---
PATH = r'C:\user\PAVLA\dataset'
N = 50  # Rozměr mřížky (50x50)
TILE_SIZE = 256
SOUBOR_CHYB = "chyby_lokalizace_jedna_dlazdice.txt"
SOUBOR_CHYB = "chyby_lokalizace_3x3.txt"

def interaktivni_kontrola_s_chybami(base_path, n, tile_size, soubor_chyb):
    img_dir = os.path.join(base_path, 'images')
    mask_dir = os.path.join(base_path, 'masks')
    
    velky_rozmer = n * tile_size
    
    # Vytvoření velkých polí (array) pro fotomapu i masku
    # ZMĚNA: Masku převedeme na RGB (3 kanály), abychom do ní mohli kreslit červené chybové dlaždice
    velka_foto = np.zeros((velky_rozmer, velky_rozmer, 3), dtype=np.uint8)
    velka_maska = np.zeros((velky_rozmer, velky_rozmer, 3), dtype=np.uint8)

    print("Skládám mapy do paměti pro interaktivní prohlížení...")

    for i in range(n):
        for j in range(n):
            filename = f"tile_{i:02d}_{j:02d}.png"
            # Výpočet pozice (osa Y roste odspodu nahoru, matice indexuje odshora dolů)
            px = i * tile_size
            py = (n - 1 - j) * tile_size
            
            p_img = os.path.join(img_dir, filename)
            p_mask = os.path.join(mask_dir, filename)

            if os.path.exists(p_img):
                tile = Image.open(p_img).convert('RGB')
                velka_foto[py:py+tile_size, px:px+tile_size] = np.array(tile)
                
            if os.path.exists(p_mask):
                mask_tile = Image.open(p_mask).convert('L')
                # Převod jednokanalové masky na RGB, aby držela krok s barevným vykreslením
                mask_rgb = np.stack([np.array(mask_tile)] * 3, axis=-1)
                velka_maska[py:py+tile_size, px:px+tile_size] = mask_rgb

    # --- DOPLNĚNÍ: NAČTENÍ A ZAKRESLENÍ CHYB ---
    chyby_pocet = 0
    if os.path.exists(soubor_chyb):
        print(f"Načítám log chyb ze souboru '{soubor_chyb}'...")
        with open(soubor_chyb, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        # Přeskočíme první řádek s hlavičkou
        for line in lines[1:]:
            line = line.strip()
            if not line: continue
            
            parts = line.split(';')
            # Načteme skutečné souřadnice (skutecna_i, skutecna_j)
            err_i = int(parts[0])
            err_j = int(parts[1])
            
            # Přepočet souřadnic chyby na pixely velké mapy
            px = err_i * tile_size
            py = (n - 1 - err_j) * tile_size
            
            # Vytvoříme červený tónovaný překryv s průhledností (mixování barev)
            # Červená barva = [255, 0, 0]. Smícháme původní obraz (65%) s čistou červenou (35%)
            alfa = 0.35
            
            # Aplikace překryvu na fotomapu
            vyřez_foto = velka_foto[py:py+tile_size, px:px+tile_size]
            cerveny_overlay_foto = (vyřez_foto * (1 - alfa) + np.array([255, 0, 0]) * alfa).astype(np.uint8)
            velka_foto[py:py+tile_size, px:px+tile_size] = cerveny_overlay_foto
            
            # Aplikace překryvu na masku
            vyřez_maska = velka_maska[py:py+tile_size, px:px+tile_size]
            cerveny_overlay_maska = (vyřez_maska * (1 - alfa) + np.array([255, 0, 0]) * alfa).astype(np.uint8)
            velka_maska[py:py+tile_size, px:px+tile_size] = cerveny_overlay_maska
            
            chyby_pocet += 1
        print(f"-> Do mapy bylo úspěšně překryto {chyby_pocet} chybových pozic.")
    else:
        print(f"⚠ Soubor chyb '{soubor_chyb}' nebyl nalezen! Mapy budou zobrazeny čisté.")

    # --- VYTVOŘENÍ PROPOJENÉHO ZOBRAZENÍ ---
    # sharex a sharey zajistí, že zoomování na jednom grafu ovládá i druhý
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)
    
    ax1.imshow(velka_foto)
    ax1.set_title(f"Fotomapa s chybami (Červené dlaždice = {chyby_pocet}x chyba)")
    
    ax2.imshow(velka_maska)
    ax2.set_title("Sémantická maska s chybami")

    # Odstranění bílých okrajů
    plt.tight_layout()
    
    print("\nNÁVOD:")
    print("1. Vyber nástroj 'Lupa' (Zoom) v dolní liště okna.")
    print("2. Táhni myší v jednom z oken pro přiblížení.")
    print("3. Druhé okno se přiblíží na úplně stejné místo a uvidíš detaily chyb.")
    
    plt.show()

if __name__ == "__main__":
    interaktivni_kontrola_s_chybami(PATH, N, TILE_SIZE, SOUBOR_CHYB)
