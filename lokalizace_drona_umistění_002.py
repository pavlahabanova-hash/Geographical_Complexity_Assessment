# micromamba activate dron_ai
# micromamba activate pavla
import os
import torch
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

def simulace_lokalizace_drona_diagnostika():
    # --- KONFIGURACE ---
    DATA_DIR = r'C:\user\PAVLA\dataset'
    MODEL_PATH = "model_cpu_checkpoint.pth"
    N = 50  # Mřížka 50x50 dlaždic
    DEVICE = "cpu"
    CONFIDENCE_THRESHOLD = 0.65 
    SOUBOR_CHYB = "chyby_lokalizace_jedna_dlazdice.txt"

    print("=== START SIMULACE VIZUÁLNÍ LOKALIZACE DRONA (1 DLAŽDICE s LOGOVÁNÍM) ===")

    # Vytvoření/přepsání souboru chyb a zápis hlavičky (formát CSV oddělený středníkem)
    with open(SOUBOR_CHYB, "w", encoding="utf-8") as f_chyby:
        f_chyby.write("skutecna_i;skutecna_j;chybna_i;chybna_j;shoda_chybna;shoda_spravna;pozice_spravneho\n")

    # 1. NAČTENÍ NEURONOVÉ SÍTĚ
    model = smp.Unet(
        encoder_name="resnet18", 
        encoder_weights=None, 
        in_channels=3, 
        classes=4
    ).to(DEVICE)
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model nenalezen: {MODEL_PATH}")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("Neuronová síť úspěšně načtena.")

    # Pouze normalizace (dron v reálu fotí tak, jak vidí)
    transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    # Pomocný převod barev mapy na indexy tříd
    def mask_to_classes(mask_img):
        mask_np = np.array(mask_img)
        new_mask = np.zeros(mask_np.shape, dtype=np.uint8)
        new_mask[mask_np == 80] = 1   # Silnice
        new_mask[mask_np == 150] = 2  # Voda
        new_mask[mask_np == 255] = 3  # Budovy
        return new_mask

    # 2. NAČTENÍ VEKTOROVÉ DATABÁZE MAPY DO PAMĚTI (Bleskové vyhledávání)
    print("Načítám vektorové masky všech 2500 poloh do databáze v RAM...")
    databaze_masek = {}
    
    for i in range(N):
        for j in range(N):
            filename = f"tile_{i:02d}_{j:02d}.png"
            mask_path = os.path.join(DATA_DIR, 'masks', filename)
            if os.path.exists(mask_path):
                mask_classes = mask_to_classes(Image.open(mask_path).convert("L"))
                databaze_masek[(i, j)] = mask_classes

    print(f"Databáze připravena. Celkem indexováno {len(databaze_masek)} referenčních poloh.")

    # Čítače statistik
    spravne_lokalizovano = 0
    spatne_lokalizovano = 0
    ztraceno_pod_prahem = 0
    celkem_testovano = 0

    print("\nSpouštím skenování fotomapy a globální map-matching...")

    with torch.no_grad():
        for i in range(N):
            for j in range(N):
                filename = f"tile_{i:02d}_{j:02d}.png"
                img_path = os.path.join(DATA_DIR, 'images', filename)
                
                # Testujeme pouze tam, kde máme k dispozici fotku i záznam v mapě
                if os.path.exists(img_path) and (i, j) in databaze_masek:
                    celkem_testovano += 1
                    skutecna_poloha = (i, j)
                    
                    # A. KROK NN: Dron vyfotí snímek a síť vygeneruje sémantickou masku
                    image = np.array(Image.open(img_path).convert("RGB"))
                    transformed = transform(image=image)
                    tensor_img = transformed['image'].unsqueeze(0).to(DEVICE)
                    logits = model(tensor_img)
                    vize_drona = torch.argmax(logits, dim=1).squeeze(0).numpy().astype(np.uint8)

                    # B. GLOBÁLNÍ VYHLEDÁVÁNÍ: Porovnáme vizi se VŠEMI maskami v Brně
                    vsechny_shody = []

                    for poloha_ref, maska_ref in databaze_masek.items():
                        procentualni_shoda = np.mean(vize_drona == maska_ref)
                        vsechny_shody.append((procentualni_shoda, poloha_ref))

                    # Seřadíme shody od nejvyšší po nejnižší pro zjištění umístění
                    vsechny_shody.sort(key=lambda x: x[0], reverse=True)
                    nejlepsi_shoda, odhadnuta_poloha = vsechny_shody[0]

                    # Spočítáme, na kolikátém místě skončila správná pozice (index + 1)
                    poradi_poloh = [poloha for shoda, poloha in vsechny_shody]
                    pozice_spravneho = poradi_poloh.index(skutecna_poloha) + 1

                    # C. VYHODNOCENÍ SHODY A LOKALIZACE
                    if nejlepsi_shoda < CONFIDENCE_THRESHOLD:
                        ztraceno_pod_prahem += 1
                        if celkem_testovano % 200 == 0:
                            print(f"Poloha {skutecna_poloha} -> Nízká spolehlivost ({nejlepsi_shoda*100:.1f} %). Dron raději hlásí: Poloha neznámá.")
                    
                    elif odhadnuta_poloha == skutecna_poloha:
                        spravne_lokalizovano += 1
                    else:
                        spatne_lokalizovano += 1
                        
                        # --- DOPLNĚNÁ DIAGNOSTIKA CHYBY ---
                        # Získáme přesnou procentuální shodu, kterou získalo SKUTEČNÉ (správné) místo
                        shoda_spravna = [shoda for shoda, poloha in vsechny_shody if poloha == skutecna_poloha][0]
                        
                        # Detailní diagnostický výpis do konzole
                        print(f"❌ CHYBA NAVIGACE na {skutecna_poloha}!")
                        print(f"   -> OMYLEM VYBRÁNO: {odhadnuta_poloha} (Shoda: {nejlepsi_shoda*100:.1f} %)")
                        print(f"   -> SPRÁVNÉ MÍSTO:   Skončilo na {pozice_spravneho}. místě v žebříčku")
                        print(f"                       Skutečná shoda tohoto místa byla: {shoda_spravna*100:.1f} %")
                        print("-" * 70)

                        # Zápis chybového řádku do textového souboru
                        with open(SOUBOR_CHYB, "a", encoding="utf-8") as f_chyby:
                            f_chyby.write(f"{skutecna_poloha[0]};{skutecna_poloha[1]};{odhadnuta_poloha[0]};{odhadnuta_poloha[1]};"
                                          f"{nejlepsi_shoda:.4f};{shoda_spravna:.4f};{pozice_spravneho}\n")

    # 3. ZÁVĚREČNÉ STATISTIKY AUTONOMNÍHO SYSTÉMU
    print("\n==================================================")
    print("    ZÁVĚREČNÁ EVALUACE VIZUÁLNÍ LOKALIZACE DRONA    ")
    print("==================================================")
    print(f"Celkem nasimulováno letových poloh: {celkem_testovano}")
    print(f"✔ ÚSPĚŠNĚ LOKALIZOVÁN (Správná poloha): {spravne_lokalizovano}x ({spravne_lokalizovano/celkem_testovano*100:.2f} %)")
    print(f"❌ CHYBNÁ LOKALIZACE (Dron vedle):       {spatne_lokalizovano}x ({spatne_lokalizovano/celkem_testovano*100:.2f} %)")
    print(f"❓ NEDOSTATEČNÁ SHODA (Ztracen):        {ztraceno_pod_prahem}x ({ztraceno_pod_prahem/celkem_testovano*100:.2f} %)")
    print(f"💾 Seznam chyb byl zapsán do:            {SOUBOR_CHYB}")
    print("==================================================")
    
    uspesnost = (spravne_lokalizovano / celkem_testovano) * 100
    if uspesnost > 85:
         print("Verdikt: Vynikající výsledek!")
    else:
        print("Verdikt: Systém občas bloudí v uniformních oblastech. Pomohlo by zvětšit unikátnost textur nebo použít silnější model.")

if __name__ == '__main__':
    simulace_lokalizace_drona_diagnostika()