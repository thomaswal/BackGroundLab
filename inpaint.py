import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import io
import os
from rembg import remove

def extraire_objet(input_path):
    """
    Utilise rembg pour extraire l'objet et créer une image sans l'objet (background only).
    
    Args:
        input_path: Chemin vers l'image originale
        
    Returns:
        Tuple (chemin de l'image sans objet, chemin du masque pour inpainting)
    """
    print(f"Extraction de l'objet depuis {input_path}...")
    
    # Charger image
    with open(input_path, 'rb') as f:
        input_image = f.read()

    # Obtenir objet uniquement
    foreground_bytes = remove(input_image)
    foreground_image = Image.open(io.BytesIO(foreground_bytes)).convert("RGBA")

    # Créer un masque à partir de l'objet (alpha)
    mask = foreground_image.split()[-1]

    # Inverser le masque pour obtenir le fond
    inverted_mask = ImageOps.invert(mask)

    # Récupérer les vraies couleurs du fond avec l'image d'origine
    original = Image.open(input_path).convert("RGBA")

    # Appliquer le masque inversé donc cacher le sujet et garder le fond
    background_only = Image.new("RGBA", original.size)
    background_only.paste(original, mask=inverted_mask)

    # Sauvegarder les résultats
    base_name = os.path.splitext(input_path)[0]
    background_path = f"{base_name}_background.png"
    mask_path = f"{base_name}_mask.png"
    
    background_only.save(background_path)
    mask.save(mask_path)
    
    print(f"Image sans objet sauvegardée sous {background_path}")
    print(f"Masque sauvegardé sous {mask_path}")
    
    return background_path, mask_path

def opencv_inpainting(image_path, mask_path, method="telea"):
    """
    Réalise l'inpainting d'une image en utilisant les algorithmes intégrés d'OpenCV.
    
    Args:
        image_path: Chemin vers l'image originale
        mask_path: Chemin vers le masque (blanc pour les zones à compléter)
        method: "telea" ou "ns" (Navier-Stokes)
    
    Returns:
        Image complétée (numpy array)
    """
    # Charger l'image et le masque
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # S'assurer que le masque est binaire (255 pour les zones à remplir)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Appliquer l'algorithme d'inpainting
    if method == "telea":
        # Algorithme de Telea (Fast Marching Method)
        result = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    else:
        # Algorithme de Navier-Stokes
        result = cv2.inpaint(image, mask, 3, cv2.INPAINT_NS)
    
    return result

def lama_inpainting(image_path, mask_path):
    """
    Utilise le modèle LaMa pour l'inpainting d'image.
    Nécessite pip install torch lama-cleaner
    """
    try:
        from lama_cleaner.model_manager import ModelManager
        from lama_cleaner.schema import InpaintRequest, HDStrategy
        
        # Charger l'image et le masque
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Initialiser le modèle LaMa
        model = ModelManager(name="lama", device="cpu")
        
        # Préparer la requête
        request = InpaintRequest(
            image=image,
            mask=mask,
            hd_strategy=HDStrategy.ORIGINAL,
            ldm_steps=20,
            ldm_sampler="plms",
        )
        
        # Réaliser l'inpainting
        result = model.process(request)
        return result
        
    except ImportError:
        print("Pour utiliser LaMa, installez les dépendances avec: pip install torch lama-cleaner")
        return None

def visualize_results(original_image, background_image, mask_image, result_image, title="Résultat"):
    """Affiche l'image originale, l'image sans objet, le masque et le résultat côte à côte."""
    # Charger les images
    if isinstance(original_image, str):
        original = cv2.imread(original_image)
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    else:
        original = original_image
        
    if isinstance(background_image, str):
        background = Image.open(background_image)
        background = np.array(background.convert("RGB"))
    else:
        background = background_image
        
    if isinstance(mask_image, str):
        mask = cv2.imread(mask_image, cv2.IMREAD_GRAYSCALE)
    else:
        mask = mask_image
        
    if isinstance(result_image, str):
        result = cv2.imread(result_image)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    else:
        if len(result_image.shape) == 3 and result_image.shape[2] == 3:
            if result_image.dtype == np.uint8 and result_image.max() > 1:
                result = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
            else:
                result = result_image
        else:
            result = cv2.cvtColor(result_image, cv2.COLOR_GRAY2RGB)
    
    # Affichage
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(original)
    axes[0].set_title("Image originale")
    axes[0].axis("off")
    
    axes[1].imshow(background)
    axes[1].set_title("Sans objet")
    axes[1].axis("off")
    
    axes[2].imshow(mask, cmap='gray')
    axes[2].set_title("Masque")
    axes[2].axis("off")
    
    axes[3].imshow(result)
    axes[3].set_title(title)
    axes[3].axis("off")
    
    plt.tight_layout()
    plt.show()

def save_result(result, output_path="image_completee.jpg"):
    """Sauvegarde l'image résultante."""
    if isinstance(result, np.ndarray):
        if result.dtype == np.uint8 and result.shape[2] == 3:
            # Convertir BGR en RGB si nécessaire
            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            Image.fromarray(result_rgb).save(output_path)
        else:
            cv2.imwrite(output_path, result)
    else:
        result.save(output_path)
    print(f"Image complétée sauvegardée avec succès: {output_path}")

def process_image(input_path, method="opencv-telea", output_path=None):
    """
    Processus complet: extraction d'objet puis inpainting.
    
    Args:
        input_path: Chemin vers l'image originale
        method: Méthode d'inpainting ("opencv-telea", "opencv-ns" ou "lama")
        output_path: Chemin pour sauvegarder l'image résultante
    """
    # Déterminer le nom du fichier de sortie
    if output_path is None:
        base_name = os.path.splitext(input_path)[0]
        output_path = f"{base_name}_complete.jpg"
    
    # 1. Extraire l'objet et obtenir l'image de fond + masque
    background_path, mask_path = extraire_objet(input_path)
    
    # 2. Appliquer l'inpainting sur les zones manquantes
    print(f"Application de l'inpainting avec la méthode {method}...")
    
    if method == "opencv-telea":
        result = opencv_inpainting(background_path, mask_path, "telea")
        title = "OpenCV Telea"
    elif method == "opencv-ns":
        result = opencv_inpainting(background_path, mask_path, "ns")
        title = "OpenCV Navier-Stokes"
    elif method == "lama":
        result = lama_inpainting(background_path, mask_path)
        title = "LaMa Inpainting"
        if result is None:
            print("L'inpainting avec LaMa a échoué. Essayez d'installer les dépendances.")
            return
    else:
        print(f"Méthode '{method}' non reconnue. Options: opencv-telea, opencv-ns, lama")
        return
    
    # 3. Visualiser et sauvegarder le résultat
    visualize_results(input_path, background_path, mask_path, result, title)
    save_result(result, output_path)
    
    return output_path

# Exemple d'utilisation
if __name__ == "__main__":
    # Chemin vers votre image originale
    input_path = './knif.jpg'  # Remplacez par le chemin de votre image
    
    # Méthode opencv-telea (plus rapide, pour les petites zones)
    #process_image(input_path, method="opencv-telea")
    
    # Autres méthodes disponibles:
    #process_image(input_path, method="opencv-ns")  # Meilleure qualité parfois, mais plus lente
    process_image(input_path, method="lama")       # Nécessite pip install torch lama-cleaner