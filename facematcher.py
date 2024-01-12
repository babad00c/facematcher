import face_recognition
from sklearn.cluster import DBSCAN
import os
import cv2
import tkinter as tk
from PIL import Image, ImageTk
from tqdm import tqdm
import pickle
import numpy as np



def save_encodings(encodings_dict, filename="encodings.pkl"):
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Gets the folder where the script is running
    file_path = os.path.join(script_dir, filename)  # Combines the folder path with filename
    with open(file_path, 'wb') as file:
        pickle.dump(encodings_dict, file)
    print(f"Encodings saved to {file_path}")
    

def read_encodings(filename="encodings.pkl"):
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Gets the folder where the script is running
    file_path = os.path.join(script_dir, filename)  # Combines the folder path with filename
    try:
        with open(file_path, 'rb') as file:
            encodings_dict = pickle.load(file)
        print(f"Encodings loaded from {file_path}")
        return encodings_dict
    except FileNotFoundError:
        print(f"No such file {file_path}")
        return {}


def load_images_from_folder(folder):
    images = {}
    for filename in os.listdir(folder):
        if filename.endswith(("jpg", "png", "jpeg")):  # Add/check other file types if needed
            path = os.path.join(folder, filename)
            image = face_recognition.load_image_file(path)
            images[path] = image
    return images


def generate_face_encodings(images_dict):
    encodings_dict = {}
    for path in tqdm(images_dict.keys(), desc="Analyzing Images"):
        # Detect all faces in the image
        # image = images_dict[path]
        # face_encodings = face_recognition.face_encodings(image)
        
        # Load your image
        image = face_recognition.load_image_file(path)

        # Find all face locations using the CNN model
        face_locations = face_recognition.face_locations(image, model="cnn")
        # Calculate the face encodings for the faces detected by the CNN model
        face_encodings = face_recognition.face_encodings(image, known_face_locations=face_locations)

        
        # Assign a unique label to each face found in the image
        for i, face_encoding in enumerate(face_encodings, start=1):
            label = f"{path}-{i}"
            encodings_dict[label] = face_encoding
    
    return encodings_dict



def cluster_faces(encodings_dict, tolerance=0.6):
    # Extracting the encodings in an array
    encodings = list(encodings_dict.values())
    
    # Using DBSCAN clustering
    clustering_model = DBSCAN(metric="euclidean", min_samples=1, eps=tolerance)
    clustering_model.fit(encodings)
    
    # Assigning labels to each image path
    clustered_dict = {}
    for path, cluster_label in zip(encodings_dict.keys(), clustering_model.labels_):
        clustered_dict[path] = cluster_label
    return clustered_dict


def identify_common_clusters(clustered_faces, images_dict1, images_dict2):
    common_clusters = {}

    # Create sets of paths for efficient look-up
    paths_set1 = set(images_dict1.keys())
    paths_set2 = set(images_dict2.keys())

    # Organize paths by cluster with their original image paths
    clusters = {}
    for labeled_path, cluster_label in clustered_faces.items():
        if cluster_label not in clusters:
            clusters[cluster_label] = []
        clusters[cluster_label].append(labeled_path)  # Store the original path

    # Determine common clusters by checking the origin of each path
    for cluster_label, paths in clusters.items():
        if any(path in paths_set1 for path in paths) and any(path in paths_set2 for path in paths):
            # For common clusters, use the labeled paths to represent individual faces
            paths = [path for path, cluster in clustered_faces.items() if cluster == cluster_label]
            common_clusters[cluster_label] = paths

    return common_clusters

def extract_faces_from_folder(image_folder_path):

    # Path to the folder containing images
    # Create a subfolder for the faces
    faces_folder_path = os.path.join(image_folder_path, 'faces')
    os.makedirs(faces_folder_path, exist_ok=True)

    # Iterate over each image file in the folder
    for image_file in tqdm(os.listdir(image_folder_path), desc='Extracting Faces'):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Load the image
            image_path = os.path.join(image_folder_path, image_file)
            image = face_recognition.load_image_file(image_path)
            
            # Find all face locations in the image
            face_locations = face_recognition.face_locations(image, model='cnn')
            
            # Loop over each face found in the image
            for face_number, face_location in enumerate(face_locations, start=1):
                # Extract the face
                top, right, bottom, left = face_location
                face_image = image[top:bottom, left:right]
                pil_image = Image.fromarray(face_image)
                
                # Save the face image to a file in the 'faces' subfolder
                face_filename = f"{os.path.splitext(image_file)[0]}-face{face_number}.jpg"
                face_path = os.path.join(faces_folder_path, face_filename)
                pil_image.save(face_path)

# Call the function with your image folder path
# generate_and_save_face_encodings('path_to_your_image_folder')
# It expects a subdolder called faces to exist
def generate_and_save_face_encodings_for_folder(folder_path):
    # Subfolder for the face images
    faces_folder_path = os.path.join(folder_path, 'faces')
    
    # Check if the faces subfolder exists
    if not os.path.exists(faces_folder_path):
        print(f"No 'faces' subfolder exists in {folder_path}")
        return

    # Iterate over each image file in the 'faces' subfolder
    for face_image_file in os.listdir(faces_folder_path):
        # Construct full file paths
        face_image_path = os.path.join(faces_folder_path, face_image_file)
        encoding_file_path = os.path.join(faces_folder_path, f"{face_image_file}-encoding.pkl")

        # Skip files that aren't images or if the encoding file already exists
        if not face_image_file.lower().endswith(('.png', '.jpg', '.jpeg')) or os.path.exists(encoding_file_path):
            continue

        # Load the face image
        face_image = face_recognition.load_image_file(face_image_path)

        # Generate the face encoding for the face image
        face_encodings = face_recognition.face_encodings(face_image)
        if face_encodings:
            # Take the first encoding if there are multiple faces
            face_encoding = face_encodings[0]

            # Save the encoding to a .pkl file
            with open(encoding_file_path, 'wb') as encoding_file:
                pickle.dump(face_encoding, encoding_file)
        else:
            print(f"No faces found in {face_image_file}, skipping.")

    print("Face encoding process complete.")


def load_face_encodings_for_folder(folder_path):
    # Subfolder for the face images and encodings
    faces_folder_path = os.path.join(folder_path, 'faces')

    # Check if the faces subfolder exists
    if not os.path.exists(faces_folder_path):
        print(f"No 'faces' subfolder exists in {folder_path}")
        return {}

    # Initialize a dictionary to hold the face encodings
    face_encodings = {}

    # Iterate over each file in the 'faces' subfolder
    for file in os.listdir(faces_folder_path):
        # Only process .pkl files
        if file.endswith('-encoding.pkl'):
            encoding_file_path = os.path.join(faces_folder_path, file)

            # Load the face encoding
            with open(encoding_file_path, 'rb') as encoding_file:
                encoding = pickle.load(encoding_file)

            # Extract the face image filename from the encoding filename
            face_image_filename = encoding_file_path.replace('-encoding.pkl', '')

            # Add the encoding to the dictionary
            face_encodings[face_image_filename] = encoding

    return face_encodings

def main():
    folder_path1 = "./known_images"
    folder_path2 = "./unknown_images"
    load_saved_encodings = False
    
    if not load_saved_encodings:
        extract_faces_from_folder(folder_path1)
        extract_faces_from_folder(folder_path2)
    generate_and_save_face_encodings_for_folder(folder_path1)
    folder1_encodings_dict = load_face_encodings_for_folder(folder_path1)
    generate_and_save_face_encodings_for_folder(folder_path2)
    folder2_encodings_dict = load_face_encodings_for_folder(folder_path2)
    
    # Step 1: Load images from both folders
    # images_dict1 = load_images_from_folder(folder_path1)
    # images_dict2 = load_images_from_folder(folder_path2)

    # Step 2: Combine the dictionaries
    combined_encodings_dict = {**folder1_encodings_dict, **folder2_encodings_dict}  # Merging two dictionaries

    # Step 4: Cluster the resulting face encodings
    clustered_faces = cluster_faces(combined_encodings_dict)

    # Identify common clusters
    common_clusters = identify_common_clusters(clustered_faces, folder1_encodings_dict, folder2_encodings_dict)
    display_clusters(common_clusters, folder2_encodings_dict)
    
    # Return or process the clustered faces as needed
    return common_clusters

def display_clusters(common_clusters, images_dict1):
    # Convert cluster dictionary to a list of tuples for easy navigation
    cluster_list = list(common_clusters.items())

    # Initialize current cluster index
    current_index = 0

    # Setup the GUI window
    window = tk.Tk()
    window.title("Face Clusters")
    
    # build image cache
    image_data_map = {}
    for cluster_number, clustered_paths in common_clusters.items():
        for path in clustered_paths:
            image_data_map[path] = np.array(face_recognition.load_image_file(path))

    # Function to update the displayed cluster
    def update_display(index):
        # Clearing previous images
        for widget in window.winfo_children():
            widget.destroy()
        cluster_label = index
        # Get current cluster
        cluster_label, paths = cluster_list[index]

        # Create a container for image widgets
        frame = tk.Frame(window)
        frame.pack()

        # Determine the grid size
        grid_size = int(len(paths)**0.5) + 1  # Simple square-ish grid

        # Load and display each face in the cluster
        for i, path in enumerate(paths):
            # Load the image as a NumPy array (assuming PIL Image)
            face_image_np = image_data_map[path]
            
            # Determine the border color based on the dictionary origin
            color = (255, 0, 0) if path in images_dict1 else (0, 0, 255)

            # Draw a border around the face
            cv2.rectangle(face_image_np, (0, 0), (face_image_np.shape[1], face_image_np.shape[0]), color, 10)
            
            # Resize the face image using OpenCV
            target_width = 250
            height, width, _ = face_image_np.shape
            scale_factor = target_width / width
            target_height = int(height * scale_factor)
            face_image_rescaled_np = cv2.resize(face_image_np, (target_width, target_height))
            face_image_rescaled = Image.fromarray(face_image_rescaled_np)
            # Convert to a format Tkinter can use
            photo = ImageTk.PhotoImage(face_image_rescaled)
            
            
            #Create and place the image in the window using grid
            label = tk.Label(frame, image=photo)
            label.image = photo  # Keep a reference!
            row, col = divmod(i, grid_size)  # Determine the position in the grid
            label.grid(row=row*2, column=col)  # Multiply row by 2 to make space for text labels

            # Create and place the text label below the image
            text_label = tk.Label(frame, text=os.path.basename(path))
            text_label.grid(row=row*2+1, column=col)  # Place it just below the image
            
            # # Detect faces and select the specific face (if available)
            # face_locations = face_recognition.face_locations(image)
            # if face_number < len(face_locations):
            #     top, right, bottom, left = face_locations[face_number]
            #     face_image_np = image[top:bottom, left:right]

            #     # Determine the border color based on the dictionary origin
            #     color = (255, 0, 0) if path in images_dict1 else (0, 0, 255)

            #     # Draw a border around the face
            #     cv2.rectangle(face_image_np, (0, 0), (face_image_np.shape[1], face_image_np.shape[0]), color, 10)

            #     # Resize the face image using OpenCV
            #     target_width = 250
            #     height, width, _ = face_image_np.shape
            #     scale_factor = target_width / width
            #     target_height = int(height * scale_factor)
            #     face_image_rescaled_np = cv2.resize(face_image_np, (target_width, target_height))

            #     # Convert back to PIL Image for displaying in Tkinter
            #     face_image_rescaled = Image.fromarray(face_image_rescaled_np)

            #     # Convert to a format Tkinter can use
            #     photo = ImageTk.PhotoImage(face_image_rescaled)

            #     #Create and place the image in the window using grid
            #     label = tk.Label(frame, image=photo)
            #     label.image = photo  # Keep a reference!
            #     row, col = divmod(i, grid_size)  # Determine the position in the grid
            #     label.grid(row=row*2, column=col)  # Multiply row by 2 to make space for text labels

            #     # Create and place the text label below the image
            #     text_label = tk.Label(frame, text=os.path.basename(path))
            #     text_label.grid(row=row*2+1, column=col)  # Place it just below the image


        # Update the window title with the cluster index
        window.title(f"Cluster {cluster_label}")

    # Function to navigate to the previous cluster
    def previous_cluster():
        nonlocal current_index
        if current_index > 0:
            current_index -= 1
            update_display(current_index)

    # Function to navigate to the next cluster
    def next_cluster():
        nonlocal current_index
        if current_index < len(cluster_list) - 1:
            current_index += 1
            update_display(current_index)


    # Bind the left arrow key to the previous_cluster function
    window.bind('<Left>', lambda event: previous_cluster())

    # Bind the right arrow key to the next_cluster function
    window.bind('<Right>', lambda event: next_cluster())

    # Buttons to navigate between clusters
    prev_button = tk.Button(window, text="<< Previous", command=previous_cluster)
    prev_button.pack(side="left")

    next_button = tk.Button(window, text="Next >>", command=next_cluster)
    next_button.pack(side="right")

    # Initially display the first cluster
    update_display(current_index)

    # Run the GUI loop
    window.mainloop()



# Example usage
# Assuming common_clusters is the output from the previous functions,
# and images_dict1 and images_dict2 are the original image dictionaries
# display_clusters(common_clusters, images_dict1, images_dict2)



if __name__ == '__main__':
    main()