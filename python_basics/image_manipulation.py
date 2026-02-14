import numpy as np
import matplotlib.pyplot as plt

def create_gradient_image():
    """CREATE A SIMPLE GRADIENT IMAGE"""
    width,height=256,256

    gradient=np.linspace(0,255,width)
    image=np.tile(gradient,(height,1))
    return image.astype(np.uint8)

def create_checkerboard(size=256,square_size=32):
    """CREATE A CHECKERBOARD PATTERN"""
    image=np.zeros((size,size),dtype=np.uint8)

    for i in range(0,size,square_size):
        for j in range(0,size,square_size):
            if ((i//square_size) + (j//square_size))%2==0:
                image[i:i+square_size,j:j+square_size]=255
    return image

def invert_image(image):
    """INVERT COLORS"""
    return 255-image

def adjust_brightness(image,factor):
    """ADJUST BRIGHTNESS"""
    adjusted=image*factor
    adjusted=np.clip(adjusted,0,255)
    return adjusted.astype(np.uint8)

def adjust_contrast(image,factor):
    """ADJUST CONTRAST"""
    mean=np.mean(image)
    adjusted=mean+factor*(image-mean)
    adjusted=np.clip(adjusted,0,255)
    return adjusted.astype(np.uint8)

def crop_image(image,x,y,width,height):
    """CROP IMAGE"""
    return image[y:y+height,x:x+width]

def rotate_90(image):
    """ROTATE IMAGE"""
    return np.rot90(image,k=-1)

def flip_horizontal(image):
    """FLIP HORIZONTAL"""
    return np.fliplr(image)

def flip_vertical(image):
    """FLIP VERTICAL"""
    return np.flipud(image)

def add_noise(image,amount=0.1):
    """ADD RANDOM NOISE"""
    noise=np.random.normal(0,amount*255,image.shape)
    noisy=image+noise
    noisy=np.clip(noisy,0,255)
    return noisy.astype(np.uint8)

def blur_simple(image,kernel_size=3):
    """SIMPLE AVERAGING BLUR"""
    height,width=image.shape
    blurred=np.zeros_like(image,dtype=float)

    k=kernel_size//2

    for i in range(k,height-k):
        for j in range(k,width-k):
            neighborhood=image[i-k:i+k+1,j-k:j+k+1]
            blurred[i,j]=np.mean(neighborhood)
    return blurred.astype(np.uint8)

def main():
    """MAIN IMAGE PROCESSOR"""
    print("="*50)
    print("IMAGE PROCESSOR")
    print("="*50)

    while True:
        print("\m-----MENU-----")
        print("1. Create gradient image")
        print("2. Create checkerboard")
        print("3. Invert colors")
        print("4. Adjust brightness")
        print("5. Adjust contrast")
        print("6. Rotate 90°")
        print("7. Flip horizontal")
        print("8. Flip vertical")
        print("9. Add noise")
        print("10. Apply blur")
        print("11. Exit")

        choice=input("\nCHOOSE (1-11):")

        if choice=="1":
            image=create_gradient_image()
            plt.imshow(image,cmap='gray')
            plt.title("GRADIENT IMAGE")
            plt.colorbar()
            plt.show()
            print("GRADIENT CREATED")
        
        elif choice=="2":
            image=create_checkerboard()
            plt.imshow(image,cmap='gray')
            plt.title("CHECKERBOARD")
            plt.show()
            print("CHECKERBOARD CREATED")
        
        elif choice in ["3", "4", "5", "6", "7", "8", "9", "10"]:
            print("\nFirst, create a base image:")
            print("1. Gradient  2. Checkerboard")
            base_choice = input("Choose: ")

            if base_choice=="1":
                image=create_gradient_image()
            else:
                image=create_checkerboard()
            
            plt.figure(figsize=(12,5))
            plt.subplot(1,2,1)
            plt.imshow(image,cmap='gray')
            plt.title("ORIGINAL")

            if choice == "3":
                result = invert_image(image)
                title = "Inverted"
            elif choice == "4":
                factor = float(input("Brightness factor (0.5=darker, 2.0=brighter): "))
                result = adjust_brightness(image, factor)
                title = f"Brightness ×{factor}"
            elif choice == "5":
                factor = float(input("Contrast factor (0.5=less, 2.0=more): "))
                result = adjust_contrast(image, factor)
                title = f"Contrast ×{factor}"
            elif choice == "6":
                result = rotate_90(image)
                title = "Rotated 90°"
            elif choice == "7":
                result = flip_horizontal(image)
                title = "Flipped Horizontally"
            elif choice == "8":
                result = flip_vertical(image)
                title = "Flipped Vertically"
            elif choice == "9":
                amount = float(input("Noise amount (0.1-0.5): "))
                result = add_noise(image, amount)
                title = f"Noise (amount={amount})"
            elif choice == "10":
                kernel = int(input("Kernel size (3, 5, 7): "))
                result = blur_simple(image, kernel)
                title = f"Blur (kernel={kernel})"

            plt.subplot(1,2,2)
            plt.imshow(result,cmap='gray')
            plt.title(title)
            plt.tight_layout()
            plt.show()

            print(f"{title} applied")
        elif choice == "11":
            print("\nGoodbye!")
            break

if __name__ == "__main__":
    # Install matplotlib if needed
    try:
        import matplotlib.pyplot as plt
    except:
        print("Installing matplotlib...")
        import os
        os.system("pip install matplotlib --break-system-packages")
        import matplotlib.pyplot as plt
    
main()    