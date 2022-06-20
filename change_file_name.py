import os

collection="C:\\Users\\username\Desktop\\sleep_drowsiness\\train\\Closed\\"
collection1="C:\\Users\\username\Desktop\\sleep_drowsiness\\train\\Open\\"

for file in os.listdir(collection):
    os.rename(collection+file,f'{collection}closed.{file}')

for file in os.listdir(collection1):
    os.rename(collection1+file,f'{collection1}open.{file}')