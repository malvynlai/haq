import argparse
import cv2
import numpy as np
from collections import Counter
from typing import List, Dict, Tuple
from pipeline import available_ingredients
from deepseek import  run_recipe_generation



def main(image_path: str):
    """Main function to run the full fridge-to-recipe pipeline."""
    print("🚀 Starting Fridge-to-Recipe Pipeline 🚀")
    print("-" * 50)

    ingredient_list = available_ingredients(image_path)
    
    if "No food items were detected" in ingredient_list:
        print("\n" + ingredient_list)
        return
        
    print("-" * 50)
    
    final_recipe = run_recipe_generation( ingredient_list[:10])
    return final_recipe
    


if __name__ == "__main__":
    image_path = 'darrien-staton-unsplash.jpg'
    
    main(image_path)