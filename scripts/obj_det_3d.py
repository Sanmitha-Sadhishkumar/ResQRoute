from panda3d.core import *
from direct.showbase.ShowBase import ShowBase
from direct.task import Task
import cv2
import numpy as np
from ultralytics import YOLO

class ModelViewer(ShowBase):
    def __init__(self):
        super().__init__()

        # Load the 3D model
        self.model = self.loader.loadModel("path/to/your/model.egg")  # Use .bam for binary format
        self.model.reparentTo(self.render)
        self.model.setScale(1, 1, 1)  # Scale the model
        self.model.setPos(0, 0, 0)  # Position the model

        # Setup camera
        self.camera.setPos(0, -10, 5)  # Set initial camera position
        self.camera.lookAt(self.model)  # Look at the model

        # Create a texture for rendering to an image
        self.texture = Texture()
        self.render_texture = self.makeTexture()
        self.render.setTexture(self.render_texture)

        # Start rendering
        self.taskMgr.add(self.render_to_image, "render_to_image")

    def makeTexture(self):
        tex = Texture()
        tex.setup2dTexture(640, 480, Texture.T_unsigned_byte, Texture.F_rgb)
        return tex

    def render_to_image(self, task):
        self.win.clear()
        self.render.setTexture(self.render_texture)
        self.render()

        # Grab the image from the window
        img = self.win.getScreenshot()
        img_np = np.array(img)

        # Convert from RGBA to BGR (OpenCV format)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)

        # Call your detection function
        detect_flammable_substances(img_np)

        return Task.cont

def detect_flammable_substances(image):
    model = YOLO('path/to/your/yolov8_model.pt')
    results = model.predict(image, conf=0.5)  # Adjust the confidence threshold
    for result in results:
        if result['label'] == 'flammable':
            print("Flammable substance detected!")
            # Process result: draw bounding boxes, etc.

if __name__ == '__main__':
    viewer = ModelViewer()
    viewer.run()
