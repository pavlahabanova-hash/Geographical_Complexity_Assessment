import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

def generate_slam_safe_map(data_dir, model_path, grid_size=50):
    DEVICE = "cpu"
    img_dir = os.path.join(data_dir, 'images')
    
    # 1. Inicializace modelu
    model = smp.Unet(encoder_name="resnet18", classes=4).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    # 2. Příprava prázdné mřížky pro výsledky (50x50 dlaždic)
    risk_grid = np.zeros((grid_size, grid_size))
    # Váhy rizika: Pozadí(0.6), Silnice(0.4), Voda(1.0), Budovy(0.2)
    class_weights = torch.tensor([0.6, 0.4, 1.0, 0.2]).to(DEVICE)

    print("Analyzuji dlaždice a skládám mapu...")

    with torch.no_grad():
        for img_name in os.listdir(img_dir):
            if not img_name.startswith('tile_'): continue
            
            # A. Získání souřadnic z názvu souboru (tile_05_10.png -> i=5, j=10)
            try:
                parts = img_name.replace('.png', '').split('_')
                i = int(parts[1]) # řádek
                j = int(parts[2]) # sloupec
            except: continue

            # B. Výpočet složitosti
            img_path = os.path.join(img_dir, img_name)
            img_raw = np.array(Image.open(img_path).convert("RGB"))
            img_tensor = transform(image=img_raw)["image"].unsqueeze(0).to(DEVICE)

            output = model(img_tensor)
            probs = torch.softmax(output, dim=1).squeeze(0)

            # Vážený index složitosti pro celou dlaždici
            complexity_map = torch.zeros((256, 256))
            for c in range(4):
                complexity_map += probs[c] * class_weights[c]
            
            avg_risk = complexity_map.mean().item()
            
            # Uložení do mřížky (pozor na i/j orientaci pro správné zobrazení)
            if i < grid_size and j < grid_size:
                risk_grid[j, i] = avg_risk # j je y-osa (sloupce), i je x-osa (řádky)

    # 3. Vykreslení výsledné mapy
    plt.figure(figsize=(10, 8))
    # Použijeme barvy 'RdYlGn_r' (červená = nebezpečí, zelená = bezpečí)
    plt.imshow(risk_grid, origin='lower', cmap='RdYlGn_r', interpolation='nearest')
    
    plt.colorbar(label='Index navigační složitosti (0 = Bezpečné, 1 = Kritické)')
    plt.title('SLAM-Safe Mapa Brna\n(pohled očima drona)')
    plt.xlabel('X (Dlaždice)')
    plt.ylabel('Y (Dlaždice)')
    
    # Uložení mapy
    plt.savefig('slam_safe_mapa_brna.png', dpi=300)
    plt.show()
    print("Mapa byla uložena jako 'slam_safe_mapa_brna.png'")

# Spuštění
generate_slam_safe_map(r'C:\user\PAVLA\dataset', 'model_cpu_checkpoint.pth')
