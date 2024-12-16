import os
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
import clip
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Global variables
current_index = 0
image_files = []
similarities = {}
folder_path = ""
tolerance = 0.5  # Default tolerance for similarity

# Load CLIP model and preprocess function
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


# Function to compute image embeddings using CLIP
def compute_embedding(image_path):
    try:
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.encode_image(image).cpu().numpy()
        return embedding / np.linalg.norm(embedding)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


# Function to calculate similarity scores between all images
def calculate_similarities():
    global similarities
    embeddings = {}

    # Compute embeddings for all images
    def process_image(img):
        try:
            return img, compute_embedding(os.path.join(folder_path, img))
        except Exception as e:
            print(f"Error processing {img}: {e}")
            return img, None

    with ThreadPoolExecutor() as executor:
        results = executor.map(process_image, image_files)
        for img, embedding in results:
            embeddings[img] = embedding

    # Compute similarity matrix
    for img1 in image_files:
        similarities[img1] = {}
        for img2 in image_files:
            if img1 != img2:
                try:
                    if embeddings[img1] is not None and embeddings[img2] is not None:
                        score = cosine_similarity(embeddings[img1], embeddings[img2])[
                            0
                        ][0]
                        similarities[img1][img2] = score
                except Exception as e:
                    similarities[img1][img2] = 0


# Function to find the top N most similar images within tolerance
def find_most_similar(image_name, top_n=3):
    if image_name in similarities:
        similar_images = sorted(
            [
                (img, score)
                for img, score in similarities[image_name].items()
                if score >= tolerance
            ],
            key=lambda x: x[1],
            reverse=True,
        )
        return similar_images[:top_n]
    return []


# GUI Functions
def load_folder():
    global folder_path, image_files, current_index
    folder_path = filedialog.askdirectory()
    if folder_path:
        image_files.clear()
        image_files.extend(
            [
                f
                for f in os.listdir(folder_path)
                if f.lower().endswith(("png", "jpg", "jpeg"))
            ]
        )
        current_index = 0

        # Add progress indicator
        progress_window = tk.Toplevel(root)
        progress_window.title("Loading...")
        progress_label = tk.Label(
            progress_window, text="Calculating similarities, please wait..."
        )
        progress_label.pack(padx=10, pady=10)

        spinner_label = tk.Label(progress_window, text="|", font=("Courier", 24))
        spinner_label.pack()

        def update_spinner():
            for char in ["|", "/", "-", "\\"]:
                spinner_label.config(text=char)
                progress_window.update()
                progress_window.after(100)

        root.update()
        for _ in range(20):
            update_spinner()
        calculate_similarities()
        progress_window.destroy()
        update_view()


def delete_image(image_name):
    global current_index
    if image_name:
        os.remove(os.path.join(folder_path, image_name))
        image_files.remove(image_name)
        current_index = min(current_index, len(image_files) - 1)
        update_view()


def update_view():
    if not image_files:
        messagebox.showinfo("Info", "No images left in the folder.")
        return

    global current_index
    current_image = image_files[current_index]
    similar_images = find_most_similar(current_image, top_n=3)

    load_image_into_label(os.path.join(folder_path, current_image), current_image_label)
    current_image_name.set(current_image)

    for idx, (sim_img, sim_score) in enumerate(similar_images):
        load_image_into_label(
            os.path.join(folder_path, sim_img), similar_image_labels[idx]
        )
        similar_image_names[idx].set(f"{sim_img} (Score: {sim_score:.2f})")

    for idx in range(len(similar_images), 3):
        similar_image_labels[idx].config(image="")
        similar_image_names[idx].set("No similar image found")


def next_image():
    global current_index
    current_index = (current_index + 1) % len(image_files)
    update_view()


def previous_image():
    global current_index
    current_index = (current_index - 1) % len(image_files)
    update_view()


def load_image_into_label(image_path, label):
    try:
        image = Image.open(image_path).resize((300, 300))
        photo = ImageTk.PhotoImage(image)
        label.config(image=photo)
        label.image = photo
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")


def adjust_tolerance(value):
    global tolerance
    tolerance = float(value)
    update_view()


# GUI Setup
root = tk.Tk()
root.title("Image Deduplication GUI")

control_frame = tk.Frame(root)
control_frame.pack(side=tk.TOP, fill=tk.X)

viewer_frame = tk.Frame(root)
viewer_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

similar_frame = tk.Frame(viewer_frame)
similar_frame.grid(row=0, column=1, padx=10, pady=10)

load_button = tk.Button(control_frame, text="Load Folder", command=load_folder)
load_button.pack(side=tk.LEFT, padx=5, pady=5)

prev_button = tk.Button(control_frame, text="Previous", command=previous_image)
prev_button.pack(side=tk.LEFT, padx=5, pady=5)

next_button = tk.Button(control_frame, text="Next", command=next_image)
next_button.pack(side=tk.LEFT, padx=5, pady=5)

tolerance_slider = tk.Scale(
    control_frame,
    from_=0.0,
    to=1.0,
    resolution=0.01,
    orient=tk.HORIZONTAL,
    label="Tolerance",
    command=adjust_tolerance,
)
tolerance_slider.set(tolerance)
tolerance_slider.pack(side=tk.LEFT, padx=5, pady=5)

current_image_label = tk.Label(viewer_frame)
current_image_label.grid(row=0, column=0, padx=10, pady=10)

similar_image_labels = [tk.Label(similar_frame) for _ in range(3)]
for idx, label in enumerate(similar_image_labels):
    label.grid(row=idx, column=0, padx=5, pady=5)

current_image_name = tk.StringVar()
similar_image_names = [tk.StringVar() for _ in range(3)]

current_name_label = tk.Label(viewer_frame, textvariable=current_image_name)
current_name_label.grid(row=1, column=0, pady=5)

for idx, sim_name_var in enumerate(similar_image_names):
    label = tk.Label(similar_frame, textvariable=sim_name_var)
    label.grid(row=idx, column=1, pady=5)

current_delete_button = tk.Button(
    viewer_frame,
    text="Delete Current",
    command=lambda: delete_image(image_files[current_index]),
)
current_delete_button.grid(row=2, column=0, pady=5)

for idx in range(3):
    button = tk.Button(
        similar_frame,
        text=f"Delete Similar {idx + 1}",
        command=lambda i=idx: delete_image(similar_image_names[i].get().split()[0]),
    )
    button.grid(row=idx, column=2, pady=5)

root.mainloop()
