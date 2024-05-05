from PIL import Image
from ultralytics import YOLO


model = YOLO(
    "E:/Dating_LLM_Sentiment_Analysis/best.onnx")


results = model(
    "E:/Dating_LLM_Sentiment_Analysis/Yolov8/images/eval/Screenshot_20240329_000235_WhatsApp.jpg")

for r in results:
    im_array = r.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()  # show image
    im.save(
        'E:\Dating_LLM_Sentiment_Analysis\Dating_LLM_sentiment_analysis\output_img.jpg')
