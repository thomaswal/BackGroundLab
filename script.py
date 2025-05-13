from rembg import remove
from PIL import Image, ImageOps
import io

# Charger image
input_path = './knif.jpg'
with open(input_path, 'rb') as f:
    input_image = f.read()

# Supprimer le fond
foreground_bytes = remove(input_image)
foreground_image = Image.open(io.BytesIO(foreground_bytes)).convert("RGBA")

# Créer un masque à partir de l'alpha (canal de transparence)
alpha_mask = foreground_image.split()[-1]

# Inverser le masque : blanc = zone à inpaint (objet à supprimer)
inverted_mask = ImageOps.invert(alpha_mask)

# Convertir le masque en niveaux de gris pur ("L") pour diffusion
final_mask = inverted_mask.convert("L")
final_mask.save("mask.png")  # Masque adapté

# Optionnel : reconstruire le fond avec le masque (pas nécessaire pour diffusion, juste visuel)
original = Image.open(input_path).convert("RGBA")
background_only = Image.new("RGBA", original.size)
background_only.paste(original, mask=inverted_mask)
background_only.save("background_only.png")
