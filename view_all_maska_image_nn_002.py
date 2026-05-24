# micromamba activate dron_ai
# micromamba activate pavla
import os
import torch
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

# --- NASTAVENÍ ---
PATH = r'C:\user\PAVLA\dataset'
MODEL_PATH = "model_cpu_checkpoint.pth"
N = 50  # Rozměr mřížky (50x50)
TILE_SIZE = 256
DEVICE = "cpu"

def interaktivni_kontrola_s_nn(base_path, model_path, n, tile_size):
    img_dir = os.path.join(base_path, 'images')
    mask_dir = os.path.join(base_path, 'masks')
    
    # 1. NAČTENÍ MODELU
    print("Načítám neuronovou síť...")
    model = smp.Unet(
        encoder_name="resnet18", 
        encoder_weights=None, 
        in_channels=3, 
        classes=4
    ).to(DEVICE)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Nelze najít soubor modelu: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # Normalizace pro model (musí být stejná jako při trénování)
    transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    velky_rozmer = n * tile_size
    # Vytvoření velkých polí pro rychlé vykreslení
    velka_foto = np.zeros((velky_rozmer, velky_rozmer, 3), dtype=np.uint8)
    velka_maska = np.zeros((velky_rozmer, velky_rozmer), dtype=np.uint8)
    velka_predikce = np.zeros((velky_rozmer, velky_rozmer), dtype=np.uint8)

    print("Skládám mapy a počítám predikce sítě (může to chvíli trvat)...")

    with torch.no_grad():
        for i in range(n):
            for j in range(n):
                filename = f"tile_{i:02d}_{j:02d}.png"
                px = i * tile_size
                py = (n - 1 - j) * tile_size 
                
                p_img = os.path.join(img_dir, filename)
                p_mask = os.path.join(mask_dir, filename)

                if os.path.exists(p_img):
                    # Načtení fotky
                    img_pil = Image.open(p_img).convert('RGB')
                    img_np = np.array(img_pil)
                    velka_foto[py:py+tile_size, px:px+tile_size] = img_np
                    
                    # Spuštění skrze neuronovou síť
                    transformed = transform(image=img_np)
                    tensor_img = transformed['image'].unsqueeze(0).to(DEVICE)
                    output = model(tensor_img)
                    
                    # Získání nejvíce pravděpodobné třídy (0, 1, 2, 3) pro každý pixel
                    pred_classes = torch.argmax(output, dim=1).squeeze(0).numpy()
                    
                    # Převod indexů tříd zpět na stupně šedi (80, 150, 255) pro vizuální shodu s maskou
                    pred_gray = np.zeros(pred_classes.shape, dtype=np.uint8)
                    pred_gray[pred_classes == 1] = 80   # Silnice
                    pred_gray[pred_classes == 2] = 150  # Voda
                    pred_gray[pred_classes == 3] = 255  # Budovy
                    
                    velka_predikce[py:py+tile_size, px:px+tile_size] = pred_gray
                    
                if os.path.exists(p_mask):
                    mask_tile = Image.open(p_mask).convert('L')
                    velka_maska[py:py+tile_size, px:px+tile_size] = np.array(mask_tile)

    # --- VYTVOŘENÍ PROPOJENÉHO ZOBRAZENÍ (3 OKNA) ---
    # sharex a sharey propojí zoomování pro všechna tři okna naráz
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
    
    ax1.imshow(velka_foto)
    ax1.set_title("1. Fotomapa (Zoomuj zde!)")
    ax1.axis('off')
    
    ax2.imshow(velka_maska, cmap='gray', vmin=0, vmax=255)
    ax2.set_title("2. Správná maska (Ideál)")
    ax2.axis('off')

    ax3.imshow(velka_predikce, cmap='gray', vmin=0, vmax=255)
    ax3.set_title("3. Predikce modelu (Pohled NN)")
    ax3.axis('off')

    plt.tight_layout()
    
    print("\nNÁVOD:")
    print("1. Vyber nástroj 'Lupa' (Zoom) v dolní liště okna.")
    print("2. Klikni a táhni myší v JAKÉMKOLIV okně.")
    print("3. Všechna tři okna se okamžitě přiblíží na totožný detail Brna.")
    
    plt.show()

if __name__ == "__main__":
    interaktivni_kontrola_s_nn(PATH, MODEL_PATH, N, TILE_SIZE)