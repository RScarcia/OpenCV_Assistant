import cv2
import numpy as np
import _collections as collection
import dialogflow
import speech_recognition as sr
from gtts import gTTS
from google.api_core.exceptions import InvalidArgument
from google.oauth2 import service_account
from threading import Thread, Lock, Event
from pygame import mixer

eyeClassifierPath = 'data\haarcascades\haarcascade_eye_tree_eyeglasses.xml'
faceClassifierPath = 'data\haarcascades\haarcascade_frontalface_alt2.xml'

mutex = Lock()
is_looking = False
count = 0


class Recognition(Thread):
    # method that applies the haar classifier to recognize the presence of the eyes in the ROI
    # identified by the haar_face function
    # it returns the coordinate of the rectangles that sorround the identified area
    def haar_eyes(self, img, equalizedImg):
        eyeClassifier = cv2.CascadeClassifier(eyeClassifierPath)

        coords = eyeClassifier.detectMultiScale(equalizedImg)

        if coords is None or len(coords) != 2:
            return None
        else:
            return coords

    # haar classifier applied on the whole frame to recognize the presence of one or more human faces
    # The biggest face in the frame will be considered as the "main face" and will be further processed
    def haar_face(self, img, equalizedImg):
        faceClassifier = cv2.CascadeClassifier(faceClassifierPath)

        rectangles = faceClassifier.detectMultiScale(equalizedImg)

        frame = None
        eqFrame = None

        # Selecting only the biggest rectangle found with the face classifier
        selected_rect = (0, 0, 0, 0)

        if len(rectangles) > 1:
            for i in rectangles:
                if i[3] > selected_rect[3]:
                    selected_rect = i
            selected_rect = np.array([i], dtype=np.int32)
        elif len(rectangles) == 1:
            selected_rect = rectangles
        else:
            return None, None

        # display the face rectangle in the frame
        for (x,y,w,h) in selected_rect:
            frame = img[y:y+h, x:x+w]
            cv2.rectangle(img,
                          (x, y),
                          (x + w, y + h),
                          (0, 255, 0), 3)
            eqFrame = equalizedImg[y:y+h, x:x+w]

        return frame, eqFrame

    def preprocess_frame(self, frame):
        grayImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        equalizedImg = cv2.equalizeHist(grayImg)
        # blurred = cv2.GaussianBlur(frame, (5, 5), 0)

        # bilateral filter that remove noise but keeps edges sharp
        bilateral = cv2.bilateralFilter(equalizedImg, 5, 75, 75, cv2.BORDER_DEFAULT)
        return bilateral

    def __init__(self):
        Thread.__init__(self)

    def run(self):
        global is_looking

        # init the camera
        cap = cv2.VideoCapture(0)

        frame_number = 20
        target_frame = 10

        # list of boolean that saves the status of the last n frame.
        # This is used to avoid false readings.
        # The face is tracked if at least "target_frame" frame are "true" in the total "frame_number"
        lastframelist = collection.deque(frame_number*[False], frame_number)

        while True:
            # get the current frame
            ret, frame = cap.read()
            preprocessed = self.preprocess_frame(frame)

            # find a list of rectangles containing the face found with haar
            faceROI, eqFaceROI = self.haar_face(frame, preprocessed)

            if faceROI is not None:
                eyes = self.haar_eyes(faceROI, eqFaceROI)
                if eyes is not None:
                    lastframelist.appendleft(True)
                    # display the eye rectangle in the face frame
                    for (x_eye, y_eye, w_eye, h_eye) in eyes:
                        cv2.rectangle(faceROI,
                                      (x_eye , y_eye ),
                                      (x_eye + w_eye, y_eye + h_eye),
                                      (255, 0, 0), 3)
                else:
                    lastframelist.appendleft(False)
            else:
                lastframelist.appendleft(False)

            if lastframelist.count(True) > target_frame:
                # the mutex is needed since this flag is used by the google assistant thread
                mutex.acquire()
                is_looking = True
                mutex.release()
            else:
                mutex.acquire()
                is_looking = False
                mutex.release()

            cv2.imshow('frame', frame)

            k = cv2.waitKey(1)
            if k == 27: #ESC
                cv2.destroyAllWindows()
                t2.stop()
                t2.join()
                break
            elif cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) < 1:
                t2.stop()
                t2.join()
                break

        cap.release()
        cv2.destroyAllWindows()


class GoogleAssistant(Thread):

    def speak(self, audioString): # text to speech
        global count

        print("response: "+audioString)
        tts = gTTS(text=audioString, lang='en')
        tts.save(f"speech{count%2}.mp3")
        mixer.init()
        mixer.music.load(f"speech{count%2}.mp3")
        mixer.music.play()
        count += 1

    def recordAudio(self):
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("Say something")
            try:
                self.audio = r.listen(source, timeout=3)
                # data = ""
                data = r.recognize_google(self.audio)
                print("you said: " + data)
                return data
            except sr.WaitTimeoutError:
                pass
            except AttributeError:
                pass
            except sr.UnknownValueError:
                print("Audio non riconosciuto")
            except sr.RequestError as e:
                print("non arriva al servizio: {0}".format(e))
        return ""

    def __init__(self):
        #Thread.__init__(self)
        super(GoogleAssistant, self).__init__()
        self._stop_event = Event()
        self.credentials = service_account.Credentials.from_service_account_file(
            'google_assistant/cvsample-9c99e-7bebe334bd60.json')

        self.DIALOGFLOW_PROJECT_ID = 'cvsample-9c99e'
        self.DIALOGFLOW_LANGUAGE_CODE = 'en-US'
        self.GOOGLE_APPLICATION_CREDENTIALS = 'google_assistant/cvsample-9c99e-7bebe334bd60.json'
        self.SESSION_ID = 'INSERT HERE YOUR GOOGLE ID'
        self.session_client = dialogflow.SessionsClient(credentials=self.credentials)
        self.session = self.session_client.session_path(self.DIALOGFLOW_PROJECT_ID, self.SESSION_ID)

    def run(self):
        global is_looking
        while True:
            if self.stopped():
                break
            text_to_be_analyzed = self.recordAudio()
            if text_to_be_analyzed == "":
                continue
            mutex.acquire()
            if is_looking:
                mutex.release()
                text_input = dialogflow.types.TextInput(text=text_to_be_analyzed,
                                                        language_code=self.DIALOGFLOW_LANGUAGE_CODE)
                query_input = dialogflow.types.QueryInput(text=text_input)
                try:
                    response = self.session_client.detect_intent(session=self.session, query_input=query_input)
                    response = response.query_result.fulfillment_text
                    self.speak(response)

                except InvalidArgument:
                    self.speak("I did not understand what you said. Can you repeat?")
                except AttributeError:
                    print("AttributeError: 'str' object has no attribute 'query_result'")

            else:
                mutex.release()
                self.speak("You have to look the camera if you want to talk with me")

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()


if __name__ == "__main__":
    try:
        t1 = Recognition()
        t2 = GoogleAssistant()
        t1.start()
        t2.start()
    except KeyboardInterrupt:
        t1.join()
        t2.stop()
        t2.join()
        exit(0)
