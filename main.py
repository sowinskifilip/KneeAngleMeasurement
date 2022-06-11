import cv2
import cameraprocessing as cp
import matplotlib.pyplot as plt

from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen, NoTransition
from kivy.core.window import Window
from kivy.clock import Clock
from datetime import datetime
from datetime import date
###koment

###*Defininig the window size*###
Window.size = (850, 750)

###* Global variables *###
zapis_ident = ""

###* Function creating txt files with given text_data and filename *###
def write_to_txt(date_time,text_data, filename):
    file = open(f"{filename}.txt", "w", encoding='utf8')
    file.write(date_time)
    file.write('\n')
    file.write(text_data)
    file.close()

    return

###* Class describing Start Window *###
class StartWindow(Screen):

    ###* Function
    def fun(self):

        return

###* Class describing Main Window *###
class MainWindow(Screen):

    def on_enter(self, *args):
        self.angles_vec = []
        self.selected_leg = 0
        self.min_angle = 180
        self.max_angle = 0
        self.summary_img_title = ''
        self.angle_range = 0

        self.updateMinMaxAngle()
        # Clock.schedule_interval(self.upadeMinMaxAngle, 1)
        return

    ###* Function triggerred with 'q' keybtn - ending the measurement *###
    def update_after_meas(self):
        name = self.ids.name_input.text
        ident = self.ids.id_input.text
        global zapis_ident
        zapis_ident = ident
        print(zapis_ident)

        # self.ids.save_label.text = (
        #     f"Patient's data has been updated: \n name: {name}, id: {ident}, angle: ..."
        # )

        return

    ###* Function triggerred with 'START' btn - starting the measurement *###
    def press_start(self):
        self.capture = cv2.VideoCapture(0)
        self.min_angle, self.max_angle, self.angles_vec, self.angle_range = cp.CameraCapture(self.capture, self.angles_vec,
                                                          self.selected_leg, self.min_angle, self.max_angle, self.angle_range)

        self.updateMinMaxAngle()
        self.plot_data()
        return

    ###* Function creating plot and saving as png in local dir *###
    def plot_data(self):
        plt.clf()
        plt.plot(self.angles_vec, 'b-o')
        plt.grid(True)
        plt.title(self.summary_img_title)
        plt.xlabel('Number of sample')
        plt.ylabel(r'Knee joint angle $[^{\circ}]$')
        plt.savefig('KneeAngle.png')
        return

    def pressed_left(self):
        self.selected_leg = 1
        self.ids.selected_leg.text = 'Leg: Left'
        self.summary_img_title = 'Left knee joint measurement'
        return

    def pressed_right(self):
        self.selected_leg = 2
        self.ids.selected_leg.text = 'Leg: Right'
        self.summary_img_title = 'Right knee joint measurement'
        return

    def updateMinMaxAngle(self, *args):
        self.ids.min_angle.text = f"MIN Angle: {self.min_angle}"
        self.ids.max_angle.text = f"MAX Angle: {self.max_angle}"
        return

###* Class describing Second Window *###
class SecondWindow(Screen):

    def on_pre_enter(self, *args):
        self.ids.summary_img.reload()

        return
    ###* Function triggerred with 'SAVE' btn creating txt file named with id and typed comments inside in local dir *###
    def press_to_save(self):
        comments = self.ids.comments_input.text
        print("DATA SAVED!!!", zapis_ident)
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        today = date.today()
        td_string = today.strftime("%d_%m_%Y")
        nazwa = zapis_ident + ' ' + td_string
        write_to_txt(dt_string,comments, nazwa)
        return


###* Loading data from kv. file *###
kv = Builder.load_file("my_box.kv")


###* Class describing Main desktop app class *###
class KneeAngleMeasurementApp(App):
    def build(self):
        sm = ScreenManager(transition=NoTransition())
        sm.add_widget(StartWindow(name="StartWindow"))
        sm.add_widget(MainWindow(name="MainWindow"))
        sm.add_widget(SecondWindow(name="SecondWindow"))
        # sm.transition()

        return sm


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    KneeAngleMeasurementApp().run()
