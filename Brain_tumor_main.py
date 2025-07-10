import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
model = load_model('brain_tumor_model.h5')

# Set image size
img_size = 128

def load_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if not file_path:
        return
    
    try:
        # Load and preprocess the image
        img = image.load_img(file_path, target_size=(img_size, img_size))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Normalize pixel values
        
        # Predict the class
        prediction = model.predict(img_array)
        print(f"Raw prediction: {prediction}")  # Debugging

        # Handle different model output shapes
        if prediction.shape[1] == 1:
            # Model with a single output neuron (Sigmoid activation)
            if prediction[0][0] > 0.5:
                label = "Tumor"
                color = "red"
            else:
                label = "No Tumor"
                color = "green"
        elif prediction.shape[1] == 2:
            # Model with two output neurons (Softmax activation)
            if prediction[0][1] > 0.5:
                label = "Tumor"
                color = "red"
            else:
                label = "No Tumor"
                color = "green"
        else:
            label = "Unknown Output Shape"
            color = "gray"
        
        # Update the result label
        result_label.config(text=f"Prediction: {label}", fg=color)
        
        # Update the displayed image
        img_display = Image.open(file_path).resize((300, 300))  # Resize for display
        img_display_with_rect = mark_tumor_region(img_display, label)
        tk_img = ImageTk.PhotoImage(img_display_with_rect)
        image_label.config(image=tk_img)
        image_label.image = tk_img
        
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

def mark_tumor_region(img, label):
    # Convert the image to RGB mode if it's not
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    width, height = img.size

    # Draw a rectangle if tumor is predicted
    if label == "Tumor":
        # For demonstration, draw a rectangle in the center of the image
        draw.rectangle([width*0.25, height*0.25, width*0.75, height*0.75], outline="red", width=5)
    
    return img

# Create the Tkinter window
window = tk.Tk()
window.title("Brain Tumor Detection")
window.geometry("800x600")  # Set the window size

# Create and place a button to load image
load_button = tk.Button(window, text="Load Image", command=load_image, font=('Helvetica', 14))
load_button.pack(pady=20)

# Create and place a label to display the image
image_label = tk.Label(window, bg="white")
image_label.pack(pady=20)

# Create and place a label to display the result
result_label = tk.Label(window, text="Prediction: ", font=('Helvetica', 18, 'bold'), bg="lightgray")
result_label.pack(pady=20)

# Run the Tkinter event loop
window.mainloop()
