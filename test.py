import torch

from PIL import Image









# Charger l'image de fond
image = Image.open("background_only.png").convert("RGB")

# Charger le masque d'inpainting
mask_image = Image.open("mask.png").convert("L")  # Doit être en niveaux de gris ("L")


from diffusers import StableDiffusionInpaintPipeline
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
)
pipe.to("cuda")
prompt = "complet background most real as possible"
#image and mask_image should be PIL images.
#The mask structure is white for inpainting and black for keeping as is
image = pipe(prompt=prompt, image=image, mask_image=mask_image).images[0]
image.save("./yellow_cat_on_park_bench.png")

