# micromamba activate dron_ai
# micromamba activate pavla
import os
import torch
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import time

def superrychla_simulace_lokalizace():
    # --- KONFIGURACE ---
    DATA_DIR = r'C:\user\PAVLA\dataset'
    MODEL_PATH = "model_cpu_checkpoint.pth"
    N = 50  # Mřížka 50x50 dlaždic
    DEVICE = "cpu"
    CONFIDENCE_THRESHOLD = 0.65 
    SOUBOR_CHYB = "chyby_lokalizace_3x3.txt"

    print("=== START DIAGNOSTICKÉ SIMULACE (OKOLÍ 3x3 s PRE-COMPUTINGEM) ===")
    START_TIME = time.time()

    # Otevřeme soubor pro zápis chyb a vyčistíme ho
    with open(SOUBOR_CHYB, "w", encoding="utf-8") as f_chyby:
        f_chyby.write("skutecna_i;skutecna_j;chybna_i;chybna_j;shoda_oblasti_chybna;shoda_oblasti_spravna;shoda_stredu_spravna\n")

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

    transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    def mask_to_classes(mask_img):
        mask_np = np.array(mask_img)
        new_mask = np.zeros(mask_np.shape, dtype=np.uint8)
        new_mask[mask_np == 80] = 1   # Silnice
        new_mask[mask_np == 150] = 2  # Voda
        new_mask[mask_np == 255] = 3  # Budovy
        return new_mask

    # 2. JEDNORÁZOVÉ NAČTENÍ VEKTOROVÉ DATABÁZE MAPY DO RAM
    print("\n[Krok 1/3] Načítám vektorové masky z disku do RAM...")
    databaze_masek = {}
    for i in range(N):
        for j in range(N):
            filename = f"tile_{i:02d}_{j:02d}.png"
            mask_path = os.path.join(DATA_DIR, 'masks', filename)
            if os.path.exists(mask_path):
                databaze_masek[(i, j)] = mask_to_classes(Image.open(mask_path).convert("L"))
    print(f"-> Vektorová mapa připravena v RAM ({len(databaze_masek)} pozic).")

    # 3. PRE-COMPUTING VIZÍ (PREDIKCÍ SÍTĚ) DO RAM
    print("\n[Krok 2/3] Projíždím celou fotomapu přes NN (Pre-computing)...")
    databaze_vizi_drona = {}
    
    with torch.no_grad():
        for i in range(N):
            for j in range(N):
                filename = f"tile_{i:02d}_{j:02d}.png"
                img_path = os.path.join(DATA_DIR, 'images', filename)
                
                if os.path.exists(img_path) and (i, j) in databaze_masek:
                    image = np.array(Image.open(img_path).convert("RGB"))
                    transformed = transform(image=image)
                    tensor_img = transformed['image'].unsqueeze(0).to(DEVICE)
                    logits = model(tensor_img)
                    vize_tile = torch.argmax(logits, dim=1).squeeze(0).numpy().astype(np.uint8)
                    databaze_vizi_drona[(i, j)] = vize_tile
    print(f"-> Všechny predikce sítě uložené v RAM ({len(databaze_vizi_drona)} vzorů).")

    # Čítače statistik
    spravne_lokalizovano = 0
    spatne_lokalizovano = 0
    ztraceno_pod_prahem = 0
    celkem_testovano = 0

    print("\n[Krok 3/3] Spouštím bleskový map-matching 3x3...")

    # Hlavní lokalizační cyklus (vynecháváme okraje kvůli 3x3 okolí)
    for i in range(1, N - 1):
        for j in range(1, N - 1):
            
            vsechno_v_ram = True
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if (i + di, j + dj) not in databaze_vizi_drona:
                        vsechno_v_ram = False
            
            if not vsechno_v_ram:
                continue

            celkem_testovano += 1
            skutecna_poloha = (i, j)

            # Načtení 9 predikovaných masek z RAM, které dron reálně nasnímal
            vize_okoli = { (di, dj): databaze_vizi_drona[(i + di, j + dj)] for di in [-1, 0, 1] for dj in [-1, 0, 1] }

            vsechny_shody = []

            for ref_i in range(1, N - 1):
                for ref_j in range(1, N - 1):
                    poloha_ref = (ref_i, ref_j)
                    
                    # Výpočet průměrné shody pro zkoumanou oblast 3x3 v databázi
                    shody_9_dlazdic = []
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            maska_ref = databaze_masek[(ref_i + di, ref_j + dj)]
                            shoda_pixelu = np.mean(vize_okoli[(di, dj)] == maska_ref)
                            shody_9_dlazdic.append(shoda_pixelu)
                    
                    prumerna_shoda_oblasti = np.mean(shody_9_dlazdic)
                    vsechny_shody.append((prumerna_shoda_oblasti, poloha_ref))

            # Seřazení výsledků od nejlepší shody po nejhorší
            vsechny_shody.sort(key=lambda x: x[0], reverse=True)
            nejlepsi_shoda, odhadnuta_poloha = vsechny_shody[0]

            # Zjištění pořadí, na kterém se umístila správná poloha
            poradi_poloh = [poloha for shoda, poloha in vsechny_shody]
            pozice_spravneho = poradi_poloh.index(skutecna_poloha) + 1

            # C. VYHODNOCENÍ SHODY A LOKALIZACE
            if nejlepsi_shoda < CONFIDENCE_THRESHOLD:
                ztraceno_pod_prahem += 1
                if celkem_testovano % 200 == 0:
                    print(f"Poloha {skutecna_poloha} -> Nízká spolehlivost oblasti ({nejlepsi_shoda*100:.1f} %). Poloha neznámá.")
            
            elif odhadnuta_poloha == skutecna_poloha:
                spravne_lokalizovano += 1
            else:
                spatne_lokalizovano += 1
                
                # --- VÝPOČET STATISTIK PRO SPRÁVNOU POLOHU ---
                # 1. Shoda CELÉ OBLASTI 3x3 na správném místě
                shoda_oblasti_spravna = [shoda for shoda, poloha in vsechny_shody if poloha == skutecna_poloha][0]
                
                # 2. Shoda ČISTĚ STŘEDOVÉHO snímku na správném místě (relativní index di=0, dj=0)
                stredova_maska_ref = databaze_masek[skutecna_poloha]
                shoda_stredu_spravna = np.mean(vize_okoli[(0, 0)] == stredova_maska_ref)
                
                # ZMĚNA: Podrobný diagnostický print
                print(f"❌ CHYBA NAVIGACE na {skutecna_poloha}!")
                print(f"   -> OMYLEM VYBRÁNO: {odhadnuta_poloha} (Shoda oblasti: {nejlepsi_shoda*100:.1f} %)")
                print(f"   -> SKUTEČNÉ MÍSTO:  Skončilo na {pozice_spravneho}. místě v žebříčku")
                print(f"                       Shoda celého okolí 3x3: {shoda_oblasti_spravna*100:.1f} %")
                print(f"                       Shoda samotného středu: {shoda_stredu_spravna*100:.1f} %")
                print("-" * 70)

                # ZMĚNA: Zápis chyby do souboru pro pozdější vizualizaci
                with open(SOUBOR_CHYB, "a", encoding="utf-8") as f_chyby:
                    f_chyby.write(f"{skutecna_poloha[0]};{skutecna_poloha[1]};{odhadnuta_poloha[0]};{odhadnuta_poloha[1]};"
                                  f"{nejlepsi_shoda:.4f};{shoda_oblasti_spravna:.4f};{shoda_stredu_spravna:.4f}\n")

    # 3. ZÁVĚREČNÉ STATISTIKY AUTONOMNÍHO SYSTÉMU
    CELKOVÝ_ČAS = time.time() - START_TIME
    print("\n==================================================")
    print("    ZÁVĚREČNÁ EVALUACE VIZUÁLNÍ LOKALIZACE DRONA    ")
    print("==================================================")
    print(f"Celkem nasimulováno letových poloh: {celkem_testovano}")
    print(f"✔ ÚSPĚŠNĚ LOKALIZOVÁN (Správná poloha): {spravne_lokalizovano}x ({spravne_lokalizovano/celkem_testovano*100:.2f} %)")
    print(f"❌ CHYBNÁ LOKALIZACE (Dron vedle):       {spatne_lokalizovano}x ({spatne_lokalizovano/celkem_testovano*100:.2f} %)")
    print(f"❓ NEDOSTATEČNÁ SHODA (Ztracen):        {ztraceno_pod_prahem}x ({ztraceno_pod_prahem/celkem_testovano*100:.2f} %)")
    print(f"💾 Chyby uloženy do souboru:             {SOUBOR_CHYB}")
    print(f"⏱ Celkový čas běhu simulace:            {CELKOVÝ_ČAS:.1f} vteřin")
    print("==================================================")

if __name__ == '__main__':
    superrychla_simulace_lokalizace()