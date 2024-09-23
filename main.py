from google.colab import drive
import os
from PIL import Image
from roboflow import Roboflow

drive.mount("/content/drive")

rf = Roboflow(api_key="pb9ussVjrHJCOmXo3axB")
project = rf.workspace("atk").project("atk-jply1")
model = project.version(5).model


folder_path = '/content/drive/MyDrive/ฟอร์มไม่มีชื่อ (File responses)/รูปผลตรวจ atk(ตั้งชื่อไฟล์เป็นชื่อตัวเอง) (File responses)'
for filename in os.listdir(folder_path):
    image_path = os.path.join(folder_path, filename)

    # Load and display the image
    img = Image.open(image_path)
    display(img)

    pred = model.predict(image_path, confidence=0, overlap=30)

    max_confidence = 0
    max_confidence_class = 'None'
    for result in pred.predictions:
        if max_confidence < result['confidence']:
            max_confidence = result['confidence']
            max_confidence_class = result['class']

    print(max_confidence_class, ':', max_confidence)
    if max_confidence > 0.5:
        it_atk = True
    else:
        it_atk = False
    if it_atk == False:
        print(filename, 'is not atk \n')
    elif it_atk and max_confidence_class == 'Pos':
        print(filename, 'is positive')