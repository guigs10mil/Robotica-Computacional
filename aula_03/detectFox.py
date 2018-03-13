import numpy as np
import cv2
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)

cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# Initiate surf detector
#surf = cv2.xfeatures2d.SIFT_create()
surf = cv2.xfeatures2d.SURF_create(hessianThreshold=5000)
madfox=cv2.imread("madfox.jpg")
kp1, des1 = surf.detectAndCompute(madfox,None)

def detect_features(img,kp1,des1,frame,frame_g):
    MIN_MATCH_COUNT = 10
   
    # find the keypoints and descriptors with SURF in each image
    
    kp2, des2 = surf.detectAndCompute(frame_g,None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    # Configura o algoritmo de casamento de features
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Tenta fazer a melhor comparacao usando o algoritmo
    matches = flann.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    circles = None
    circles=cv2.HoughCircles(frame_g,cv2.HOUGH_GRADIENT,2,60,param1=200,param2=100,minRadius=5,maxRadius=60)

    if circles is not None:
        circles = np.uint16(np.around(circles))

        for i in circles[0,:]:
            print(i)
            # draw the outer circle
            # cv2.circle(img, center, radius, color[, thickness[, lineType[, shift]]])
            cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(frame,(i[0],i[1]),2,(0,0,255),3)

    if len(good)>MIN_MATCH_COUNT and circles is not None:
        # Separa os bons matches na origem e no destino
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)


        # Tenta achar uma trasformacao composta de rotacao, translacao e escala que situe uma imagem na outra
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        h,w = img.shape[0],img.shape[1]
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

        # Transforma os pontos da imagem origem para onde estao na imagem destino
        dst = cv2.perspectiveTransform(pts,M)

        # Desenha as linhas
        frame = cv2.polylines(frame,[np.int32(dst)],True,255,3, cv2.LINE_AA)
        
    else:
        print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
        matchesMask = None

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                    singlePointColor = None,
                    matchesMask = matchesMask, # draw only inliers
                    flags = 2)
    #frame=drawMatches(img,kp1,frame_g,kp2,good[:20])
    
    # HoughCircles - detects circles using the Hough Method. For an explanation of
    # param1 and param2 please see an explanation here http://www.pyimagesearch.com/2014/07/21/detecting-circles-images-using-opencv-hough-circles/
 

    return frame


while(True):
    #print(timer)
    # Capture frame-by-frame
    #print("Novo frame")
    ret, frame = cap.read()

    frame_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frame_gray=cv2.medianBlur(frame_gray,5)
    #testar o bilateral filtering
  

    #More drawing functions @ http://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html
    frame=detect_features(madfox,kp1, des1,frame,frame_gray)
    
    # Display the resulting frame
    cv2.imshow('original',frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #print("No circles were found")
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()