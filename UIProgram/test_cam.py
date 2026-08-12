import cv2

# 0 通常是 Mac 电脑自带的前置摄像头
# 1 或者 2 通常就是你连上的 iPhone 摄像头
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if ret:
        cv2.imshow('iPhone Camera Test', frame)
    # 按键盘上的 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()