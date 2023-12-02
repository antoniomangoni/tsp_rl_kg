from tkinter import filedialog
from tkinter import Tk
from PIL import Image

def process_tree_image(file_path, output_path, new_size=(600, 1000)):
    # Step 1: Open an image file
    with Image.open(file_path) as img:
        # Step 2: Convert image to RGBA
        img = img.convert("RGBA")
        
        # Step 3: Find white background and make it transparent
        data = img.getdata()
        new_data = []
        for item in data:
            # Change all white (also shades of whites)
            # pixels to transparent
            if item[0] in list(range(230, 256)):
                new_data.append((item[0], item[1], item[2], 0))
            else:
                new_data.append(item)
        img.putdata(new_data)
        
        # Step 4: Resize the image
        # img = img.resize(new_size, Image.ANTIALIAS)
        
        # Step 5: Save the image
        img.save(output_path, 'PNG')

if __name__ == "__main__":
    root = Tk()
    root.withdraw()  # Hide the root window

    file_path = filedialog.askopenfilename(title="Select the PNG Image")
    if file_path:
        output_path = filedialog.asksaveasfilename(title="Save the Processed Image", defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if output_path:
            process_tree_image(file_path, output_path)
        else:
            print("Output file path not specified!")
    else:
        print("Input file path not specified!")
