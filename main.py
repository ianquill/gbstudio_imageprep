import cv2
import numpy as np
from sklearn.cluster import KMeans
from tkinter import filedialog, Button, Entry, Label, Canvas, StringVar
from tkinter import Tk
from PIL import Image, ImageTk

# Function to split image into 8x8 squares
def split_into_blocks(image):
    blocks = []
    height, width = image.shape[:2]
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            block = image[i:i + 8, j:j + 8]
            if block.shape[0] == 8 and block.shape[1] == 8: 
                blocks.append(block)
    return blocks

# Function to flatten blocks for KMeans
def flatten_blocks(blocks):
    block_vectors = []
    for block in blocks:
        block_vectors.append(block.flatten())
    return np.array(block_vectors)

# Function to replace image blocks with cluster centroid
def replace_blocks(image, kmeans, blocks):
    height, width = image.shape[:2]
    count = 0
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            block = image[i:i + 8, j:j + 8]
            if block.shape[0] == 8 and block.shape[1] == 8: 
                image[i:i + 8, j:j + 8] = kmeans.cluster_centers_[kmeans.labels_[count]].reshape(8, 8, 3)
                count += 1
    return image

def process_image(image_path, output_size, num_clusters, output_filename):
    # Load the image
    image = cv2.imread(image_path)

    # Convert to RGB color space (OpenCV loads images in BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize the output image (center crop)
    height, width = image.shape[:2]
    startx = width//2 - output_size[0]//2
    starty = height//2 - output_size[1]//2 
    image = image[starty:starty+output_size[1], startx:startx+output_size[0]]

    # Split image into blocks
    blocks = split_into_blocks(image)

    # Flatten blocks
    block_vectors = flatten_blocks(blocks)

    # Apply KMeans
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(block_vectors)

    # Replace blocks with cluster centroid
    image = replace_blocks(image, kmeans, blocks)


    # Convert back to BGR color space and save
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_filename, image)

    # Display the output image
    output_image = Image.open(output_filename)
    output_image.thumbnail((500, 500))
    photo = ImageTk.PhotoImage(output_image)
    output_label.config(image=photo)
    output_label.image = photo

def load_image():
    global original_image, file_path
    file_path = filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("png files","*.png"),("all files","*.*")))
    original_image = Image.open(file_path)
    original_image.thumbnail((500, 500))
    photo = ImageTk.PhotoImage(original_image)
    image_label.config(image=photo)
    image_label.image = photo
    resolution_label.config(text=f"Resolution: {original_image.size[0]} x {original_image.size[1]}")

def start_processing():
    output_size = (int(x_entry.get()), int(y_entry.get()))
    num_clusters = int(num_tiles_entry.get())
    output_filename = output_filename_entry.get()
    process_image(file_path, output_size, num_clusters, output_filename)

if __name__ == '__main__':
    root = Tk()

    # Button to load the image
    Button(root, text="Load Image", command=load_image).pack()

    # Label to display the image
    image_label = Label(root)
    image_label.pack()

    # Label to display the resolution
    resolution_label = Label(root, text="Resolution: N/A")
    resolution_label.pack()

    # Entry boxes and labels to set the output size
    x_entry = Entry(root)
    x_entry.pack()
    x_entry.insert(0, "1920")  # Default value
    Label(root, text="Output Width").pack()
    
    y_entry = Entry(root)
    y_entry.pack()
    y_entry.insert(0, "1080")  # Default value
    Label(root, text="Output Height").pack()

    # Entry box and label to set the number of unique tiles
    num_tiles_entry = Entry(root)
    num_tiles_entry.pack()
    num_tiles_entry.insert(0, "192")  # Default value
    Label(root, text="Number of unique tiles").pack()

    # Entry box to set the output filename
    output_filename_entry = Entry(root)
    output_filename_entry.pack()
    output_filename_entry.insert(0, "output.png")  # Default value
    Label(root, text="Output filename").pack()

    # Button to start processing the image
    Button(root, text="Start Processing", command=start_processing).pack()

    # Label to display the output image
    output_label = Label(root)
    output_label.pack()

    root.mainloop()