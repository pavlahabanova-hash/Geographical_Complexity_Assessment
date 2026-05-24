# micromamba activate pavla
import os
import tkinter as tk
from tkinter import ttk, messagebox
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

class GISProhlizecOptimalizovany:
    def __init__(self, root, slozka_dat):
        self.root = root
        self.root.title("GIS Prohlížeč - Optimalizovaný pro rychlost")
        self.root.geometry("1400x900")

        self.slozka_dat = slozka_dat
        self.vrstvy_data = {}      
        self.stav_checkboxu = {}    

        # --- GUI Rozvržení ---
        self.sidebar = tk.Frame(self.root, width=300, bg="#f8f9fa", padx=10, pady=10)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)

        self.map_container = tk.Frame(self.root, bg="white")
        self.map_container.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        tk.Label(self.sidebar, text="Vrstvy v paměti", font=("Arial", 11, "bold"), bg="#f8f9fa").pack(pady=(0, 10))

        # Seznam s posuvníkem
        self.canvas_sidebar = tk.Canvas(self.sidebar, bg="#f8f9fa", highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.sidebar, orient="vertical", command=self.canvas_sidebar.yview)
        self.scrollable_frame = tk.Frame(self.canvas_sidebar, bg="#f8f9fa")

        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas_sidebar.configure(scrollregion=self.canvas_sidebar.bbox("all")))
        self.canvas_sidebar.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas_sidebar.configure(yscrollcommand=self.scrollbar.set)
        self.canvas_sidebar.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.btn_refresh = tk.Button(self.sidebar, text="AKTUALIZOVAT POHLED", 
                                     command=self.vykresli_mapu, bg="#28a745", fg="white", 
                                     font=("Arial", 10, "bold"), pady=10)
        self.btn_refresh.pack(side=tk.BOTTOM, fill=tk.X, pady=10)

        # --- Matplotlib ---
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.patch.set_facecolor('#f0f0f0')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.map_container)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.map_container)
        self.toolbar.update()

        self.canvas.mpl_connect('scroll_event', self.zoom_koleckem)

        # Načtení dat do RAM hned při startu
        self.nacti_vse_do_ram()

    def nacti_vse_do_ram(self):
        """Načte všechny SHP soubory do paměti pro bleskovou odezvu."""
        if not os.path.exists(self.slozka_dat):
            messagebox.showerror("Chyba", "Složka s daty nebyla nalezena!")
            return
        test_oblast_stupne = (16.50, 49.10, 16.75, 49.25)
        bbox_metry = (-599100, -1160200, -598850, -1159950)

        soubory = [f for f in os.listdir(self.slozka_dat) if f.endswith('.shp')]
        
        for soubor in soubory:
            nazev = soubor.replace(".shp", "")
            try:
                # gdf = gpd.read_file(os.path.join(self.slozka_dat, soubor),bbox= test_oblast_stupne)
                # gdf = gpd.read_file(os.path.join(self.slozka_dat, soubor),bbox= test_oblast_stupne)
                gdf = gpd.read_file(os.path.join(self.slozka_dat, soubor))
                if gdf.crs != "EPSG:5514":
                    gdf = gdf.to_crs("EPSG:5514")
                
                # Vytvoření prostorového indexu pro bleskové ořezy (cx)
                gdf.sindex 
                self.vrstvy_data[nazev] = gdf
                
                var = tk.BooleanVar(value=False)
                self.stav_checkboxu[nazev] = var
                chk = tk.Checkbutton(self.scrollable_frame, text=nazev, variable=var, bg="#f8f9fa")
                chk.pack(anchor="w")
            except Exception as e:
                print(f"Chyba u {soubor}: {e}")

    def zoom_koleckem(self, event):
        base_scale = 1.3
        cur_xlim, cur_ylim = self.ax.get_xlim(), self.ax.get_ylim()
        if event.xdata is None or event.ydata is None: return

        scale_factor = 1/base_scale if event.button == 'up' else base_scale
        
        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
        
        rel_x = (cur_xlim[1] - event.xdata) / (cur_xlim[1] - cur_xlim[0])
        rel_y = (cur_ylim[1] - event.ydata) / (cur_ylim[1] - cur_ylim[0])

        self.ax.set_xlim([event.xdata - new_width * (1 - rel_x), event.xdata + new_width * rel_x])
        self.ax.set_ylim([event.ydata - new_height * (1 - rel_y), event.ydata + new_height * rel_y])
        self.canvas.draw()

    def vykresli_mapu(self):
        """Efektivní vykreslení s využitím prostorového ořezu a zjednodušení."""
        # Zapamatujeme si aktuální výřez, abychom ořízli data
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        
        self.ax.clear()
        self.ax.set_facecolor("#dae1e7")

        # Logika pořadí (spodní vrstvy -> horní vrstvy)
        priority = {"water": 1, "natural": 2, "landuse": 3, "admin": 4, "roads": 5, "build": 6}
        
        aktivni_nazvy = [n for n, v in self.stav_checkboxu.items() if v.get()]
        # Seřadíme podle priority, aby budovy nebyly pod lesem
        aktivni_nazvy.sort(key=lambda x: next((v for k, v in priority.items() if k in x.lower()), 99))

        for nazev in aktivni_nazvy:
            gdf = self.vrstvy_data[nazev]
            
            # 1. RYCHLÝ OŘEZ: Vykreslujeme jen to, co je v okně
            # (Funguje pouze pokud už uživatel aspoň jednou zoomoval)
            if xlim != (0.0, 1.0):
                gdf_plot = gdf.cx[xlim[0]:xlim[1], ylim[0]:ylim[1]]
            else:
                gdf_plot = gdf

            if gdf_plot.empty: continue

            # 2. AUTOMATICKÉ ZJEDNODUŠENÍ: Pokud je dat moc, zjednodušíme geometrii o 1-2 metry
            if len(gdf_plot) > 10000:
                gdf_plot = gdf_plot.copy()
                gdf_plot.geometry = gdf_plot.simplify(1.5)

            # Nastavení barev
            barva, sirka = "#555555", 0.5
            if "water" in nazev.lower(): barva, sirka = "#74a9cf", 0.8
            elif "build" in nazev.lower(): barva, sirka = "#e31a1c", 0.1
            elif "road" in nazev.lower(): barva, sirka = "#ffffff", 0.4
            elif "natural" in nazev.lower(): barva, sirka = "#addd8e", 0.1
            
            gdf_plot.plot(ax=self.ax, color=barva, linewidth=sirka, alpha=0.9)

        # Obnovíme limity po smazání ax.clear()
        if xlim != (0.0, 1.0):
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)
            
        self.ax.grid(True, color='white', linestyle='-', linewidth=0.5, alpha=0.5)
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    # CESTA K TVÝM DATŮM
    MOJE_CESTA = r'C:\user\PAVLA\mapa'
    
    app = GISProhlizecOptimalizovany(root, MOJE_CESTA)
    root.mainloop()