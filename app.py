import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode, RTCConfiguration
import av

# Import class model của bạn
from model import FaceEmotionCNN 

# -----------------------------------------------------------
# 1. CẤU HÌNH & LOAD MODEL
# -----------------------------------------------------------
CLASSES = ['angry', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'suprise']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'final_model.pth'

# Cấu hình STUN Server cho WebRTC (để chạy online)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Load Face Detection (Haar Cascade)
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

emotion_model = load_model()

# Transform cho model (Grayscale -> 75x75 -> Tensor)
val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((75, 75)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# -----------------------------------------------------------
# 2. XỬ LÝ WEBCAM REALTIME
# -----------------------------------------------------------
class EmotionProcessor(VideoTransformerBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            if roi_gray.size == 0: continue

            try:
                roi_tensor = val_transform(roi_gray).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    outputs = emotion_model(roi_tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=1)[0]
                    conf, pred_idx = torch.max(probs, 0)
                
                emotion_label = CLASSES[pred_idx.item()]
                confidence = conf.item() * 100

                # Vẽ khung
                color = (0, 255, 0)
                if emotion_label in ['angry', 'fear', 'disgust', 'sad']:
                    color = (0, 0, 255)
                
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                cv2.putText(img, f"{emotion_label} ({confidence:.1f}%)", (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            except Exception:
                pass
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# -----------------------------------------------------------
# 3. GIAO DIỆN CHÍNH
# -----------------------------------------------------------
st.title("🎥 AI Emotion Recognition")
app_mode = st.sidebar.selectbox("Chọn chế độ", ["Webcam Realtime", "Upload Ảnh"])

if app_mode == "Webcam Realtime":
    st.write("Sử dụng Webcam để nhận diện cảm xúc theo thời gian thực.")
    if emotion_model is not None:
        ctx = webrtc_streamer(
            key="realtime-emotion", 
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=EmotionProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True
        )
        st.info("Nhấn 'START' và cho phép trình duyệt truy cập Camera.")

elif app_mode == "Upload Ảnh":
    st.write("Upload ảnh chứa khuôn mặt để nhận diện.")
    uploaded_file = st.file_uploader("Chọn ảnh...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Load ảnh và hiển thị
        image_pil = Image.open(uploaded_file).convert('RGB')
        st.image(image_pil, caption='Ảnh gốc', width=400)
        
        if st.button('🔍 Phân tích cảm xúc'):
            with st.spinner('Đang tìm khuôn mặt và phân tích...'):
                # Chuyển đổi sang format OpenCV để tìm mặt
                img_cv = np.array(image_pil) 
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR) # Convert RGB to BGR
                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                
                # Tìm khuôn mặt
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                if len(faces) == 0:
                    st.warning("⚠️ Không tìm thấy khuôn mặt nào trong ảnh! Đang thử phân tích toàn bộ ảnh...")
                    # Nếu không thấy mặt, thử đưa cả ảnh vào (resize về 75x75)
                    face_roi = gray 
                    display_img = img_cv
                else:
                    st.success(f"Đã tìm thấy {len(faces)} khuôn mặt.")
                    # Lấy khuôn mặt to nhất (hoặc đầu tiên) để xử lý
                    (x, y, w, h) = faces[0]
                    face_roi = gray[y:y+h, x:x+w]
                    
                    # Vẽ khung lên ảnh để hiển thị kết quả
                    display_img = img_cv.copy()
                    cv2.rectangle(display_img, (x, y), (x+w, y+h), (0, 255, 0), 3)

                # Dự đoán
                try:
                    roi_tensor = val_transform(face_roi).unsqueeze(0).to(DEVICE)
                    
                    with torch.no_grad():
                        outputs = emotion_model(roi_tensor)
                        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
                    
                    # Kết quả
                    conf, pred_idx = torch.max(probs, 0)
                    pred_label = CLASSES[pred_idx.item()]
                    
                    # Hiển thị
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB), caption="Vị trí khuôn mặt", width=300)
                    with col2:
                        st.metric(label="Cảm xúc dự đoán", value=pred_label.upper())
                        st.progress(int(conf.item() * 100))
                        st.write(f"Độ tin cậy: **{conf.item()*100:.2f}%**")
                    
                    # Biểu đồ chi tiết
                    st.write("---")
                    probs_dict = {name: float(p) for name, p in zip(CLASSES, probs)}
                    st.bar_chart(probs_dict)
                    
                except Exception as e:
                    st.error(f"Lỗi khi xử lý: {e}")