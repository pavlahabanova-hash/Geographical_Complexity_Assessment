# micromamba activate dron_ai
# micromamba activate pavla
import os
import multiprocessing
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

# --- SYSTÉMOVÉ NASTAVENÍ ---
os.environ["CUDA_VISIBLE_DEVICES"] = "" # Vynucení CPU
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # Prevence pádu knihovny na Windows

# --- DEFINICE DATASETU ---
class DroneDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Bezpečné načtení párů (obrázek + maska)
        img_dir = os.path.join(root_dir, 'images')
        mask_dir = os.path.join(root_dir, 'masks')
        
        if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
            raise FileNotFoundError(f"Složky 'images' nebo 'masks' nebyly nalezeny v {root_dir}")

        img_files = set(os.listdir(img_dir))
        mask_files = set(os.listdir(mask_dir))
        self.images = sorted(list(img_files.intersection(mask_files)))
        self.images = [f for f in self.images if f.endswith('.png')]
        
        print(f"--- DATASET ---")
        print(f"Celkem nalezeno {len(self.images)} validních párů na disku.")

    def __len__(self):
        return len(self.images)

    def mask_to_classes(self, mask):
        mask_np = np.array(mask)
        new_mask = np.zeros(mask_np.shape, dtype=np.int64)
        new_mask[mask_np == 80] = 1   # Silnice
        new_mask[mask_np == 150] = 2  # Voda
        new_mask[mask_np == 255] = 3  # Budovy
        return new_mask

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.root_dir, 'images', img_name)
        mask_path = os.path.join(self.root_dir, 'masks', img_name)
        
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = self.mask_to_classes(Image.open(mask_path).convert("L"))

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        return image, mask

# --- HLAVNÍ TRÉNOVACÍ FUNKCE ---
def run_cpu_training():
    # --- KONFIGURACE ---
    DATA_DIR = r'C:\user\PAVLA\dataset'
    MODEL_PATH = "model_cpu_checkpoint.pth"
    BATCH_SIZE = 64
    ADDITIONAL_EPOCHS = 100  
    DEVICE = "cpu"

    # --- PŘÍPRAVA INTERAKTIVNÍHO GRAFU ---
    plt.ion() # Zapnutí interaktivního módu
    fig, ax = plt.subplots(figsize=(10, 6))
    
    train_loss_history = []
    val_loss_history = []
    epochs_axis = []
    
    ax.set_title("Průběh učení modelu (Trénovací vs Validační ztráta)")
    ax.set_xlabel("Epocha")
    ax.set_ylabel("Ztráta (Loss)")
    ax.grid(True, linestyle='--', alpha=0.6)
    
    line_train, = ax.plot([], [], 'b-o', linewidth=2, markersize=5, label='Trénovací Loss')
    line_val, = ax.plot([], [], 'r-s', linewidth=2, markersize=5, label='Validační Loss')
    ax.legend()

    # --- TRANSFORMACE ---
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),  
        A.RandomRotate90(p=0.5), 
        A.ColorJitter(brightness=0.2, contrast=0.2, p=0.5), 
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    # --- NAČTENÍ A ROZDĚLENÍ DAT (80:20) ---
    full_dataset = DroneDataset(DATA_DIR, transform=transform)
    
    val_size = int(len(full_dataset) * 0.2)  
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Rozdělení dat -> Trénovací sada: {len(train_dataset)} obrázků | Validační sada: {len(val_dataset)} obrázků")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # --- MODEL ---
    model = smp.Unet(
        encoder_name="resnet18", 
        encoder_weights="imagenet", 
        in_channels=3, 
        classes=4
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-7)
    loss_fn = nn.CrossEntropyLoss()

    # PŘIDÁNO: LR Scheduler (pokud val_loss 5 epoch neklesne, zmenší lr o 90 %)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1)

    # --- NAČTENÍ PŘEDCHOZÍHO STAVU ---
    start_epoch = 0  
    if os.path.exists(MODEL_PATH):
        print(f"Načítám váhy z: {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        start_epoch = 0 
        print(f"Pokračuji od epochy {start_epoch + 1}")

    print("\n--- TRÉNOVÁNÍ STARTUJE ---")

    try:
        best_val_loss = float('inf') # Nekonečno jako startovní bod
        for epoch in range(start_epoch, start_epoch + ADDITIONAL_EPOCHS):
            
            # === FÁZE A: TRÉNOVÁNÍ ===
            model.train()
            epoch_train_loss = 0
            
            for batch_idx, (data, targets) in enumerate(train_loader):
                data, targets = data.to(DEVICE), targets.to(DEVICE).long()

                optimizer.zero_grad()
                predictions = model(data)
                loss = loss_fn(predictions, targets)
                loss.backward()
                optimizer.step()

                epoch_train_loss += loss.item()
                
                if batch_idx % 40 == 0:
                    print(f"Epocha {epoch+1} [{batch_idx}/{len(train_loader)}] -> Aktuální Trénovací Loss: {loss.item():.4f}")

            avg_train_loss = epoch_train_loss / len(train_loader)
            train_loss_history.append(avg_train_loss)
            
            # === FÁZE B: VALIDACE ===
            model.eval()
            epoch_val_loss = 0
            
            with torch.no_grad(): 
                for data, targets in val_loader:
                    data, targets = data.to(DEVICE), targets.to(DEVICE).long()
                    predictions = model(data)
                    loss = loss_fn(predictions, targets)
                    epoch_val_loss += loss.item()
            
            avg_val_loss = epoch_val_loss / len(val_loader)
            val_loss_history.append(avg_val_loss)
            
            epochs_axis.append(epoch + 1)
            
            # PŘIDÁNO: Aktualizace scheduleru na základě výsledku validace
            scheduler.step(avg_val_loss)
            
            # UPRAVENO: Kontrolní výpis s aktuálním stavem Learning Rate (LR)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"===> KONEC EPOCHY {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr}")

            # AKTUALIZACE OBOU KŘIVEK V GRAFU
            line_train.set_data(epochs_axis, train_loss_history)
            line_val.set_data(epochs_axis, val_loss_history)
            
            ax.relim()
            ax.autoscale_view()
            plt.draw()
            plt.pause(0.1) 
            
            # Uložení stavu (POUZE POKUD JE LEPŠÍ NEŽ DOSAVADNÍ MINIMUM)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), MODEL_PATH)
                print(f"🔥 Nový nejlepší model uložen s Val Loss: {avg_val_loss:.4f}")
                
            fig.savefig('vyvoj_ztraty.png')

    except KeyboardInterrupt:
        print("\nTrénování přerušeno uživatelem. Ukládám aktuální stav...")
        torch.save(model.state_dict(), MODEL_PATH)

    print("\nTrénování dokončeno.")
    plt.ioff() 
    plt.show() 

# --- SPOUŠTĚČ ---
if __name__ == '__main__':
    multiprocessing.freeze_support()
    run_cpu_training()
