import cv2
import pandas as pd
import numpy as np
def cut_video(input_path, output_path, start_time, end_time):
    print(input_path)
    video = cv2.VideoCapture(input_path)

    
    fps = video.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    # Set the video's position to the start frame
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Create a VideoWriter object to write the trimmed video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(output_path, fourcc, fps, (int(video.get(3)), int(video.get(4))))

    # Read and write frames until the end frame is reached
    frame_counter = start_frame
    while frame_counter <= end_frame:
        ret, frame = video.read()
        if not ret:
            break

        output_video.write(frame)
        frame_counter += 1

    video.release()
    output_video.release()

    # print("Video cut and saved successfully.")




path='Dataset\\dataset_1000.csv'
df=pd.read_csv(path)
labels=df['emotion']
s=df['start_time']
e=df['end_time']

filename=df['file_name']
for i in range(len(s)):
    input_path = 'data\\video\\'+filename[i]+'.avi'
    output_path = 'splittedData/splittedVideo/'+labels[i]+'_'+str(i)+'_.avi'
    start_time = round(s[i])
    end_time = round(e[i]) # End time in seconds

    cut_video(input_path, output_path, start_time, end_time)
