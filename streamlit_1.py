import cv2
import streamlit as st


# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(r'D:\DL_Projects\emotion\haarcascade_frontalface_default.xml')


def main():
    st.title("Emotion Detection App")
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = cv2.imread(uploaded_file.name)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
            # Calculate positions of facial features for emotion analysis
            roi_gray = gray[y:y + h, x:x + w]
    
            # Calculate average pixel intensity in the region of eyes and mouth
            avg_intensity = int(roi_gray[roi_gray.shape[0] // 2, roi_gray.shape[1] // 2])
    
            # Analyze average intensity to predict emotions
            if avg_intensity < 100:
                emotion = "Sad"
            elif avg_intensity < 140:
                emotion = "Neutral"
            else:
                emotion = "Happy"
    
            # Display emotion label near the face
            cv2.putText(image, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        st.image(image, channels="BGR")
    
if __name__ == "__main__":
    main()
