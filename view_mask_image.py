import tkinter as tk
from tkinter import ttk
import os
from PIL import Image, ImageTk

class DatasetBrowser:
    def __init__(self, root, base_path):
        self.root = root
        self.root.title("Dataset Browser - Kontrola Foto vs Maska")
        self.base_path = base_path
        
        # Cesty ke složkám
        self.img_dir = os.path.join(base_path, 'images')
        self.mask_dir = os.path.join(base_path, 'masks')
        
        # Nastavení indexů (předpokládáme 40x40)
        self.i = 0
        self.j = 0
        
        self.setup_ui()
        self.update_display()

    def setup_ui(self):
        # Horní panel pro ovládání
        ctrl_frame = ttk.Frame(self.root, padding="10")
        ctrl_frame.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(ctrl_frame, text="Řádek (i):").pack(side=tk.LEFT)
        self.ent_i = ttk.Entry(ctrl_frame, width=5)
        self.ent_i.insert(0, "0")
        self.ent_i.pack(side=tk.LEFT, padx=5)

        ttk.Label(ctrl_frame, text="Sloupec (j):").pack(side=tk.LEFT)
        self.ent_j = ttk.Entry(ctrl_frame, width=5)
        self.ent_j.insert(0, "0")
        self.ent_j.pack(side=tk.LEFT, padx=5)

        ttk.Button(ctrl_frame, text="Zobrazit", command=self.go_to).pack(side=tk.LEFT, padx=10)
        ttk.Button(ctrl_frame, text="< Předchozí", command=self.prev_tile).pack(side=tk.LEFT)
        ttk.Button(ctrl_frame, text="Další >", command=self.next_tile).pack(side=tk.LEFT)

        self.lbl_status = ttk.Label(ctrl_frame, text="")
        self.lbl_status.pack(side=tk.RIGHT)

        # Hlavní plocha pro obrázky
        img_frame = ttk.Frame(self.root, padding="10")
        img_frame.pack(side=tk.TOP)

        self.panel_img = ttk.Label(img_frame)
        self.panel_img.pack(side=tk.LEFT, padx=5)

        self.panel_mask = ttk.Label(img_frame)
        self.panel_mask.pack(side=tk.LEFT, padx=5)

    def update_display(self):
        filename = f"tile_{self.i:02d}_{self.j:02d}.png"
        img_path = os.path.join(self.img_dir, filename)
        mask_path = os.path.join(self.mask_dir, filename)

        if os.path.exists(img_path) and os.path.exists(mask_path):
            # Načtení a zvětšení pro lepší viditelnost (z 256 na 512)
            img = Image.open(img_path).resize((512, 512), Image.NEAREST)
            mask = Image.open(mask_path).resize((512, 512), Image.NEAREST)

            self.tk_img = ImageTk.PhotoImage(img)
            self.tk_mask = ImageTk.PhotoImage(mask)

            self.panel_img.configure(image=self.tk_img)
            self.panel_mask.configure(image=self.tk_mask)
            self.lbl_status.configure(text=f"Zobrazeno: {filename}", foreground="black")
        else:
            self.lbl_status.configure(text=f"Chyba: {filename} neexistuje!", foreground="red")

    def go_to(self):
        try:
            self.i = int(self.ent_i.get())
            self.j = int(self.ent_j.get())
            self.update_display()
        except ValueError:
            pass

    def next_tile(self):
        self.j += 1
        if self.j >= 40:
            self.j = 0
            self.i += 1
        self.update_inputs()
        self.update_display()

    def prev_tile(self):
        self.j -= 1
        if self.j < 0:
            self.j = 39
            self.i -= 1
        self.update_inputs()
        self.update_display()

    def update_inputs(self):
        self.ent_i.delete(0, tk.END)
        self.ent_i.insert(0, str(self.i))
        self.ent_j.delete(0, tk.END)
        self.ent_j.insert(0, str(self.j))

if __name__ == "__main__":
    root = tk.Tk()
    # ZDE ZADEJ CESTU KE SVÉMU DATASETU
    path = r'C:\user\PAVLA\dataset'
    app = DatasetBrowser(root, path)
    root.mainloop()
