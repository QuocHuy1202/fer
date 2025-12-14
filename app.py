import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import av

# Import class model của bạn
from model import FaceEmotionCNN 

# 1. Cấu hình & Load Model
CLASSES = ['angry', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'suprise']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'final_model.pth'

# Load Face Detection (Haar Cascade có sẵn trong cv2)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@st.cache_resource
def load_model():
    model = FaceEmotionCNN(num_classes=len(CLASSES), in_channels=1)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except FileNotFoundError:
        st.error(f"Không tìm thấy file {MODEL_PATH}. Hãy chắc chắn bạn đã để file model cùng thư mục.")
        return None
    model.to(DEVICE)
    model.eval()
    return model

# Load model một lần duy nhất
emotion_model = load_model()

# Định nghĩa Transform giống lúc train (Grayscale -> Resize -> Tensor)
val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((75, 75)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 2. Class xử lý video Real-time
class EmotionProcessor(VideoTransformerBase):
    def recv(self, frame):
        # Chuyển frame từ định dạng av sang numpy array (OpenCV)
        img = frame.to_ndarray(format="bgr24")

        # 1. Chuyển sang ảnh xám để tìm khuôn mặt
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. Phát hiện khuôn mặt
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # 3. Cắt vùng khuôn mặt (ROI - Region of Interest)
            roi_gray = gray[y:y+h, x:x+w]
            
            # Xử lý ngoại lệ nếu mặt quá nhỏ
            if roi_gray.size == 0:
                continue

            # 4. Tiền xử lý để đưa vào model (dùng transform đã định nghĩa)
            try:
                roi_tensor = val_transform(roi_gray).unsqueeze(0).to(DEVICE)

                # 5. Dự đoán
                with torch.no_grad():
                    outputs = emotion_model(roi_tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=1)[0]
                    conf, pred_idx = torch.max(probs, 0)
                    
                emotion_label = CLASSES[pred_idx.item()]
                confidence = conf.item() * 100

                # 6. Vẽ khung và nhãn lên hình gốc
                color = (0, 255, 0) # Màu xanh lá
                if emotion_label in ['angry', 'fear', 'disgust', 'sad']:
                    color = (0, 0, 255) # Màu đỏ cho cảm xúc tiêu cực
                
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                cv2.putText(img, f"{emotion_label} ({confidence:.1f}%)", (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            except Exception as e:
                pass # Bỏ qua lỗi xử lý frame để stream không bị ngắt

        # Trả về frame đã vẽ hình
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# 3. Giao diện Streamlit
st.title("🎥 Real-time Emotion Recognition")
st.write("Sử dụng Webcam để nhận diện cảm xúc theo thời gian thực.")

# Thêm tuỳ chọn (Sidebar)
app_mode = st.sidebar.selectbox("Chọn chế độ", ["Webcam Realtime", "Upload Ảnh"])

if app_mode == "Webcam Realtime":
    if emotion_model is not None:
        ctx = webrtc_streamer(
            key="example", 
            mode=WebRtcMode.SENDRECV, # Quan trọng: Chế độ gửi và nhận
            video_processor_factory=EmotionProcessor,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False}
        )
        st.info("Nhấn 'START' để bật Camera. Hãy đảm bảo đủ ánh sáng để nhận diện tốt nhất.")

elif app_mode == "Upload Ảnh":
    # (Giữ lại code upload ảnh cũ của bạn ở đây nếu muốn)
    uploaded_file = st.file_uploader("Chọn ảnh...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Ảnh gốc', width=300)
        
        # Code xử lý ảnh tĩnh (cần thêm phần detect face cho ảnh tĩnh để chính xác hơn)
        # Ở đây demo đơn giản đưa cả ảnh vào (như code cũ) hoặc bạn có thể update logic detect face vào đây.
        if st.button('Dự đoán'):
            st.title("🎭 Facial Emotion Recognition")
            st.write("Upload một bức ảnh khuôn mặt để AI đoán cảm xúc nhé!")

            uploaded_file = st.file_uploader("Chọn ảnh...", type=["jpg", "png", "jpeg"],key="upload_image_1")

            if uploaded_file is not None:
                # Hiển thị ảnh gốc
                image = Image.open(uploaded_file)
                st.image(image, caption='Ảnh đã upload', width=300)
                
                if st.button('Dự đoán cảm xúc'):
                    with st.spinner('Đang phân tích...'):
                        try:
                            # Load model và xử lý ảnh
                            model = load_model()
                            img_tensor = process_image(image)
                            
                            # Dự đoán
                            with torch.no_grad():
                                outputs = model(img_tensor)
                                # Dùng Softmax để ra xác suất %
                                probs = torch.nn.functional.softmax(outputs, dim=1)[0]
                            
                            # Lấy kết quả cao nhất
                            conf, pred_idx = torch.max(probs, 0)
                            pred_label = CLASSES[pred_idx.item()]
                            
                            # Hiển thị kết quả
                            st.success(f"Dự đoán: **{pred_label.upper()}** ({conf.item()*100:.2f}%)")
                            
                            # Vẽ biểu đồ xác suất các lớp khác
                            st.write("---")
                            st.write("Chi tiết xác suất:")
                            probs_dict = {class_name: float(prob) for class_name, prob in zip(CLASSES, probs)}
                            st.bar_chart(probs_dict)
                            
                        except Exception as e:
                            st.error(f"Có lỗi xảy ra: {e}")