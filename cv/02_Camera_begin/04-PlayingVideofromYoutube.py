import cv2
 
########### 카메라 대신 youtube영상으로 대체 ############
# pafy 패키지 : Youtube의 메타 데이터를 수집/검색하거나 다운로드
# pip install pafy, pip install youtube-dl

import pafy
url = 'https://www.youtube.com/watch?v=xLD8oWRmlAE'
video = pafy.new(url)
print('title = ', video.title)              # 제목
print('video.rating = ', video.rating)      # 별점 정보
print('video.duration = ', video.duration)  # Play 시간
 
# get best resolution of a specific format
# set format out of(mp4, webm, flv or 3gp)
best = video.getbest(preftype='mp4')
print('best.resolution', best.resolution)

cap=cv2.VideoCapture(best.url)
#########################################################
# frame 사이즈
frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
              int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
              int(cap.get(cv2.CAP_PROP_FPS)))
print('frame_size =', frame_size)


# 코덱 설정하기
#fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # ('D', 'I', 'V', 'X')
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# 이미지 저장하기 위한 영상 파일 생성
out1 = cv2.VideoWriter('./data/record0.mp4',fourcc, 20.0, 
                                (frame_size[0], frame_size[1]))
out2 = cv2.VideoWriter('./data/record1.mp4',fourcc, 20.0, 
                                (frame_size[0], frame_size[1]), isColor=False)

while True:
    retval, frame = cap.read()  # 영상을 한 frame씩 읽어오기
    if not retval:
        break   
        
    out1.write(frame)	        # 영상 파일에 저장   
    
    # 이미지 컬러 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    out2.write(gray)	        # 영상 파일에 저장        
    
    cv2.imshow('frame',frame)	# 이미지 보여주기
    cv2.imshow('gray',gray)      
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
out1.release()
out2.release()
cv2.destroyAllWindows()

