import os
import requests
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import box
from rasterio import features
from rasterio.transform import from_bounds
from pyproj import Transformer
from PIL import Image

# --- KONFIGURACE ---
SLOZKA_DAT = r'C:\user\PAVLA\mapa'
VYSTUPNI_SLOZKA = r'C:\user\PAVLA\dataset'
# Střed Brna jako výchozí bod
X_START, Y_START = -604000, -1165000 
X_START, Y_START = -616500, -1165000 
X_START, Y_START = -605250, -1165000 
VELIKOST_DLAZDICE = 250  # metry
MRIZKA_ROZMER = 50       # 50x50 dlaždic

# Cesty k SHP souborům
SHP_BUDOVY = os.path.join(SLOZKA_DAT, 'gis_osm_buildings_a_free_1.shp')
SHP_VODA_P = os.path.join(SLOZKA_DAT, 'gis_osm_water_a_free_1.shp')
SHP_VODA_L = os.path.join(SLOZKA_DAT, 'gis_osm_waterways_free_1.shp')
SHP_SILNICE = os.path.join(SLOZKA_DAT, 'gis_osm_roads_free_1.shp')

# --- PŘÍPRAVA PROSTŘEDÍ ---
os.makedirs(os.path.join(VYSTUPNI_SLOZKA, 'images'), exist_ok=True)
os.makedirs(os.path.join(VYSTUPNI_SLOZKA, 'masks'), exist_ok=True)

def nacti_vrstvu(cesta):
    if os.path.exists(cesta):
        print(f"Načítám {os.path.basename(cesta)}...")
        gdf = gpd.read_file(cesta)
        return gdf.to_crs("EPSG:5514")
    return None

# 1. NAČTENÍ VŠEHO DO RAM
budovy_all = nacti_vrstvu(SHP_BUDOVY)
voda_p_all = nacti_vrstvu(SHP_VODA_P)
voda_l_all = nacti_vrstvu(SHP_VODA_L)
silnice_all = nacti_vrstvu(SHP_SILNICE)

def stahni_foto(bbox, cesta):
    wms_base = "https://ags.cuzk.gov.cz/arcgis1/services/ORTOFOTO/MapServer/WMSServer"
    params = {
        "SERVICE": "WMS", "VERSION": "1.1.1", "REQUEST": "GetMap",
        "LAYERS": "0", "SRS": "EPSG:5514",
        "BBOX": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "WIDTH": "256", "HEIGHT": "256", "FORMAT": "image/png", "STYLES": ""
    }
    try:
        r = requests.get(wms_base, params=params, timeout=15)
        if r.status_code == 200:
            with open(cesta, 'wb') as f: f.write(r.content)
            return True
    except: pass
    return False

# --- HLAVNÍ CYKLUS GENEROVÁNÍ ---
print(f"Začínám generovat {MRIZKA_ROZMER**2} dlaždic...")

for i in range(MRIZKA_ROZMER):
    for j in range(MRIZKA_ROZMER):
        idx = i * MRIZKA_ROZMER + j
        
        x_min = X_START + (i * VELIKOST_DLAZDICE)
        y_min = Y_START + (j * VELIKOST_DLAZDICE)
        x_max = x_min + VELIKOST_DLAZDICE
        y_max = y_min + VELIKOST_DLAZDICE
        bbox = (x_min, y_min, x_max, y_max)
        
        jmeno = f"tile_{i:02d}_{j:02d}"
        cesta_img = os.path.join(VYSTUPNI_SLOZKA, 'images', f"{jmeno}.png")
        cesta_mask = os.path.join(VYSTUPNI_SLOZKA, 'masks', f"{jmeno}.png")
        
        # A. STÁHNOUT FOTKU
        # Pokud fotka už existuje, můžeme ji přeskočit, nebo ji stáhnout znovu
        if not os.path.exists(cesta_img):
            if not stahni_foto(bbox, cesta_img):
                continue

        # B. PŘÍPRAVA MASKY
        shapes = []
        
        # 1. Silnice (80)
        if silnice_all is not None:
            vyrez = silnice_all.cx[x_min:x_max, y_min:y_max]
            if not vyrez.empty:
                for geom in vyrez.geometry.buffer(4):
                    shapes.append((geom, 80))

        # 2. Voda (150)
        for layer in [voda_p_all, voda_l_all]:
            if layer is not None:
                vyrez = layer.cx[x_min:x_max, y_min:y_max]
                if not vyrez.empty:
                    for geom in vyrez.geometry:
                        if geom.geom_type == 'LineString':
                            shapes.append((geom.buffer(6), 150))
                        else:
                            shapes.append((geom, 150))

        # 3. Budovy (255)
        if budovy_all is not None:
            vyrez = budovy_all.cx[x_min:x_max, y_min:y_max]
            if not vyrez.empty:
                for geom in vyrez.geometry:
                    shapes.append((geom, 255))

        # C. RASTERIZACE A ULOŽENÍ
        if shapes:
            mask_array = features.rasterize(
                shapes,
                out_shape=(256, 256),
                transform=from_bounds(*bbox, 256, 256),
                fill=0,
                all_touched=True
            )
        else:
            # Pokud není co kreslit, vytvoříme prázdné pole
            mask_array = np.zeros((256, 256), dtype=np.uint8)
        
        # Uložíme jako regulérní šedotónový PNG pomocí PIL
        mask_png = Image.fromarray(mask_array.astype(np.uint8))
        mask_png.save(cesta_mask)
        
        if idx % 50 == 0:
            print(f"Postup: {idx} / {MRIZKA_ROZMER**2} dlaždic hotovo...")

print("Dataset je kompletní a opravený!")