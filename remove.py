import os

dir_name = "C:/Users/vijit/OneDrive/Desktop/ANPR/license/annotations"
test = os.listdir(dir_name)

for item in test:
    if item.endswith(".xml"):
        os.remove(os.path.join(dir_name, item))