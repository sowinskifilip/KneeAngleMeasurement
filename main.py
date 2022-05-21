import cameraprocessing as cp
import cv2
import numpy as np
import matplotlib.pyplot as plt

from kivy.config import Config
Config.set('modules', 'monitor', '')
Config.set('graphics', 'maxfps', '30')

from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.clock import Clock
from kivy.graphics.texture import Texture


# def main():
#     cp.CameraCapture()
#
#     return

class MyApp(App):

    def build(self):
        layout = BoxLayout(orientation='vertical')
        self.image = Image()
        # layout.add_widget(self.image)
        # layout.add_widget(Button(
        #     text='CLICK HERE',
        #     pos_hint={'center_x': .5, 'center_y': .5},
        #     size_hint=(None,None))
        # )
        self.angles_vec = []
        self.capture = cv2.VideoCapture(0)
        # Clock.schedule_interval(self.updateCam, 1.0/30.0)
        cp.CameraCapture(self.capture, self.angles_vec)
        self.updateImage(self.angles_vec)

        layout.add_widget(self.image)
        layout.add_widget(Button(
            text='CLICK HERE',
            pos_hint={'center_x': .5, 'center_y': .5},
            size_hint=(None, None))
        )

        return layout

    def updateCam(self, *args):

        frame, angle = cp.CameraCapture(self.capture)
        # ret, frame = self.capture.read()

        self.angles_vec.put(angle)

        self.image_frame = frame
        buffer = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
        self.image.texture = texture


    def updateImage(self, *args):
        # plt.plot(self.angles_vec)
        # plt.savefig('KneeAngle.png')
        self.image = Image(source='KneeAngle.png')




if __name__ == '__main__':
    MyApp().run()


